#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--interval <SECONDS>]

Description:
- Scan bg-deploy-*/workflow.yaml in current directory and read metadata.generateName as name prefixes
- Continuously watch the latest Argo Workflow for each prefix until all reach terminal phase
- Exit code is 0 if all Succeeded; 1 if any is Failed/Error

Example:
  $0 --interval 60
EOF
}

INTERVAL=300

while [ "$#" -gt 0 ]; do
  case "$1" in
    --interval)
      shift
      INTERVAL="${1:-10}"
      ;;
    --interval=*)
      INTERVAL="${1#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Collect generateName prefixes
prefixes=()
for dir in "$BASE_DIR"/bg-deploy-*; do
  [ -d "$dir" ] || continue
  wf_yaml="$dir/workflow.yaml"
  if [ -f "$wf_yaml" ]; then
    # Extract like: generateName: bg-deploy-xxx-
    prefix=$(awk '/^[[:space:]]*generateName:[[:space:]]*/{print $2; exit}' "$wf_yaml" || true)
    if [ -n "${prefix:-}" ]; then
      prefixes+=("$prefix")
    fi
  fi
done

if [ ${#prefixes[@]} -eq 0 ]; then
  echo "No prefixes found (missing bg-deploy-*/workflow.yaml or generateName)." >&2
  exit 1
fi

echo "Monitoring prefixes: ${prefixes[*]}"
echo

# Check if a phase is terminal
is_terminal_phase() {
  case "$1" in
    Succeeded|Failed|Error)
      return 0 ;;
    *)
      return 1 ;;
  esac
}

# Fetch all workflows: name, creationTimestamp, phase (3 columns per line)
fetch_all_wf_lines() {
  kubectl get wf \
    -o jsonpath='{range .items[*]}{.metadata.name} {.metadata.creationTimestamp} {.status.phase}{"\n"}{end}' 2>/dev/null || true
}

# Select the latest workflow for a given prefix (max creation time)
select_latest_for_prefix() {
  local prefix="$1"
  local lines="$2"
  # Filter by prefix and sort by 2nd column (ISO8601), take last line
  echo "$lines" | awk -v s="$prefix" '$1 ~ ("^" s) {print $0}' | sort -k2 | tail -n1
}

# Main loop
while true; do
  now_ts=$(date '+%Y-%m-%d %H:%M:%S')
  lines=$(fetch_all_wf_lines)

  all_found=true
  all_terminal=true
  any_failed=false

  printf '[%s] Current status\n' "$now_ts"
  printf '%-36s %-32s %-12s\n' "PREFIX" "WORKFLOW_NAME" "PHASE"
  printf '%-36s %-32s %-12s\n' "------------------------------------" "--------------------------------" "------------"

  for prefix in "${prefixes[@]}"; do
    latest_line=$(select_latest_for_prefix "$prefix" "$lines" || true)
    if [ -z "${latest_line:-}" ]; then
      all_found=false
      printf '%-36s %-32s %-12s\n' "$prefix" "-" "NotCreated"
      all_terminal=false
      continue
    fi

    # Parse line: name ts phase
    wf_name=$(echo "$latest_line" | awk '{print $1}')
    wf_phase=$(echo "$latest_line" | awk '{print $3}')
    [ -n "${wf_phase:-}" ] || wf_phase="Unknown"

    printf '%-36s %-32s %-12s\n' "$prefix" "$wf_name" "$wf_phase"

    if ! is_terminal_phase "$wf_phase"; then
      all_terminal=false
    elif [ "$wf_phase" != "Succeeded" ]; then
      any_failed=true
    fi
  done

  echo

  if $all_terminal; then
    if $any_failed; then
      echo "Completed: some workflows are Failed/Error."
      exit 1
    else
      echo "Completed: all workflows Succeeded."
      exit 0
    fi
  fi

  sleep "$INTERVAL"
done

