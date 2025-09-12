#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_agent_server.sh [options]

Optional:
  --replicas <N>          Replicas (no change by default)
  --chat-model <NAME>     Model name (override CHAT_MODEL)
  --chat-base-url <URL>   Model server URL (override CHAT_BASE_URL)
  --chat-api-key <KEY>    Write chat api-key into Secret (override stringData.api-key)
  --law-searcher-url <U>  Law searcher service URL (override LAW_SEARCHER_URL)
  --case-searcher-url <U> Case searcher service URL (override CASE_SEARCHER_URL)
  --reranker-url <U>      Reranker service URL (override RERANKER_URL)
  -h, --help              Show this help

Example:
  ./run_agent_server.sh \
    --chat-base-url=http://vllm.default.svc.cluster.local:8000/v1 \
    --law-searcher-url=https://host/mcp/law-searcher/mcp/ \
    --case-searcher-url=https://host/mcp/case-searcher/mcp/ \
    --reranker-url=https://host/mcp/reranker/mcp/
USAGE
}

REPLICAS=""
MODEL_NAME=""
MODEL_SERVER=""
LAW_SEARCHER_URL=""
CASE_SEARCHER_URL=""
RERANKER_URL=""
API_KEY=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --replicas=*) REPLICAS="${1#*=}" ;;
    --replicas) shift; REPLICAS="${1:-}" ;;
    --chat-model=*) MODEL_NAME="${1#*=}" ;;
    --chat-model) shift; MODEL_NAME="${1:-}" ;;
    --chat-base-url=*) MODEL_SERVER="${1#*=}" ;;
    --chat-base-url) shift; MODEL_SERVER="${1:-}" ;;
    --chat-api-key=*) API_KEY="${1#*=}" ;;
    --chat-api-key) shift; API_KEY="${1:-}" ;;
    --law-searcher-url=*) LAW_SEARCHER_URL="${1#*=}" ;;
    --law-searcher-url) shift; LAW_SEARCHER_URL="${1:-}" ;;
    --case-searcher-url=*) CASE_SEARCHER_URL="${1#*=}" ;;
    --case-searcher-url) shift; CASE_SEARCHER_URL="${1:-}" ;;
    --reranker-url=*) RERANKER_URL="${1#*=}" ;;
    --reranker-url) shift; RERANKER_URL="${1:-}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done


BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_FILE="$BASE_DIR/k8s.yaml"

if [ ! -f "$TARGET_FILE" ]; then
  echo "File not found: $TARGET_FILE" >&2
  exit 1
fi

escape_sed() {
  # Escape special chars (& and |) used in sed replacement. Delimiter is |
  printf '%s' "$1" | sed -e 's/[&|]/\\&/g'
}

update_line_simple() {
  # $1: regex key (e.g. image: or replicas:)
  # $2: replacement after key (raw, without quotes)
  local key="$1"; shift
  local new_value="$1"; shift || true
  local v
  v="$(escape_sed "$new_value")"
  local tmp_file="$TARGET_FILE.tmp"
  sed -E "s|^([[:space:]]*)${key}[[:space:]]*.*$|\\1${key} ${v}|" "$TARGET_FILE" > "$tmp_file" && mv "$tmp_file" "$TARGET_FILE"
}

update_env_value() {
  # $1: ENV name, $2: new value (will be quoted)
  local env_name="$1"; shift
  local env_value="$1"; shift || true
  local v
  v="$(escape_sed "$env_value")"
  local tmp_file="$TARGET_FILE.tmp"
  sed -E "/name:[[:space:]]*${env_name}/{n;s|^([[:space:]]*)value:[[:space:]]*.*$|\\1value: \"${v}\"|}" "$TARGET_FILE" > "$tmp_file" && mv "$tmp_file" "$TARGET_FILE"
}

update_secret_api_key() {
  # $1: new API Key, override stringData.api-key
  local key_value
  key_value="$(escape_sed "$1")"
  local tmp_file="$TARGET_FILE.tmp"
  sed -E "s|^([[:space:]]*)api-key:[[:space:]]*.*$|\\1api-key: \"${key_value}\"|" "$TARGET_FILE" > "$tmp_file" && mv "$tmp_file" "$TARGET_FILE"
}

echo "Updating file: $TARGET_FILE"

# 1) Replicas
if [ -n "$REPLICAS" ]; then
  echo "- Set replicas: $REPLICAS"
  update_line_simple "replicas:" "$REPLICAS"
fi

# 2) Environment variables
if [ -n "$MODEL_NAME" ]; then
  echo "- Set CHAT_MODEL: $MODEL_NAME"
  update_env_value "CHAT_MODEL" "$MODEL_NAME"
fi

if [ -n "$MODEL_SERVER" ]; then
  echo "- Set CHAT_BASE_URL: $MODEL_SERVER"
  update_env_value "CHAT_BASE_URL" "$MODEL_SERVER"
fi

if [ -n "$LAW_SEARCHER_URL" ]; then
  echo "- Set LAW_SEARCHER_URL: $LAW_SEARCHER_URL"
  update_env_value "LAW_SEARCHER_URL" "$LAW_SEARCHER_URL"
fi

if [ -n "$CASE_SEARCHER_URL" ]; then
  echo "- Set CASE_SEARCHER_URL: $CASE_SEARCHER_URL"
  update_env_value "CASE_SEARCHER_URL" "$CASE_SEARCHER_URL"
fi

if [ -n "$RERANKER_URL" ]; then
  echo "- Set RERANKER_URL: $RERANKER_URL"
  update_env_value "RERANKER_URL" "$RERANKER_URL"
fi

if [ -n "$API_KEY" ]; then
  echo "- Update Secret stringData.api-key"
  update_secret_api_key "$API_KEY"
fi

echo "Applying Kubernetes resources..."
echo "kubectl apply -f $TARGET_FILE"
kubectl apply -f "$TARGET_FILE"

echo "Agent deployment completed."
