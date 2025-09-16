#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 --milvus-uri=<URI> --embedding-base-url=<URL> --chat-base-url=<URL> [--git-repo-civil-case=<URL>] [--git-repo-criminal-case=<URL>] [--git-repo-cn-law=<URL>] [--is-llm-extract=<true|false>] [--llm-workers=<N>] [--gpu-resource=<nvidia.com/gpu|enflame.com/gcu>]"
}

MILVUS_URI=""
EMBEDDING_BASE_URL=""
CHAT_BASE_URL=""
GIT_REPO_CIVIL_CASE=""
GIT_REPO_CRIMINAL_CASE=""
GIT_REPO_CN_LAW=""
IS_LLM_EXTRACT=""
LLM_WORKERS=""
GPU_RESOURCE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --milvus-uri=*)
      MILVUS_URI="${1#*=}"
      ;;
    --milvus-uri)
      shift
      MILVUS_URI="${1:-}"
      ;;
    --embedding-base-url=*)
      EMBEDDING_BASE_URL="${1#*=}"
      ;;
    --embedding-base-url)
      shift
      EMBEDDING_BASE_URL="${1:-}"
      ;;
    --chat-base-url=*)
      CHAT_BASE_URL="${1#*=}"
      ;;
    --chat-base-url)
      shift
      CHAT_BASE_URL="${1:-}"
      ;;
    --git-repo-civil-case=*)
      GIT_REPO_CIVIL_CASE="${1#*=}"
      ;;
    --git-repo-civil-case)
      shift
      GIT_REPO_CIVIL_CASE="${1:-}"
      ;;
    --git-repo-criminal-case=*)
      GIT_REPO_CRIMINAL_CASE="${1#*=}"
      ;;
    --git-repo-criminal-case)
      shift
      GIT_REPO_CRIMINAL_CASE="${1:-}"
      ;;
    --git-repo-cn-law=*)
      GIT_REPO_CN_LAW="${1#*=}"
      ;;
    --git-repo-cn-law)
      shift
      GIT_REPO_CN_LAW="${1:-}"
      ;;
    --is-llm-extract=*)
      IS_LLM_EXTRACT="${1#*=}"
      ;;
    --is-llm-extract)
      shift
      IS_LLM_EXTRACT="${1:-}"
      ;;
    --llm-workers=*)
      LLM_WORKERS="${1#*=}"
      ;;
    --llm-workers)
      shift
      LLM_WORKERS="${1:-}"
      ;;
    --gpu-resource=*)
      GPU_RESOURCE="${1#*=}"
      ;;
    --gpu-resource)
      shift
      GPU_RESOURCE="${1:-}"
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

if [ -z "$MILVUS_URI" ] || [ -z "$EMBEDDING_BASE_URL" ] || [ -z "$CHAT_BASE_URL" ]; then
  echo "Error: Missing required arguments" >&2
  usage
  exit 1
fi

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Iterate all bg-deploy-* directories
for dir in "$BASE_DIR"/bg-deploy-*; do
  [ -d "$dir" ] || continue
  echo "Processing directory: $dir"

  cm_file="$dir/configmap.yaml"
  if [ -f "$cm_file" ]; then
    echo "Updating $cm_file"
    # Replace only the value parts inside quotes to avoid changing keys/formatting
    # Preserve existing indentation and other keys
    tmp_file="$cm_file.tmp"
    awk -v milvus="$MILVUS_URI" -v emb="$EMBEDDING_BASE_URL" -v chat="$CHAT_BASE_URL" -v isllm="$IS_LLM_EXTRACT" -v workers="$LLM_WORKERS" -v dname="$(basename "$dir")" '
      function replace_val(line, key, val,   prefix, quote){
        # match patterns like: key="..."
        prefix = "\\s*" key "=\""
        if (line ~ key "=\"") {
          sub(key "=\"[^\"]*\"", key "=\"" val "\"", line)
        }
        return line
      }
      {
        $0 = replace_val($0, "MILVUS_URI", milvus)
        $0 = replace_val($0, "EMBEDDING_BASE_URL", emb)
        $0 = replace_val($0, "CHAT_BASE_URL", chat)
        # Only inject IS_LLM_EXTRACT and LLM_WORKERS for civil/criminal dirs when provided
        if ((dname == "bg-deploy-civil-case" || dname == "bg-deploy-criminal-case")) {
          if (isllm != "") { $0 = replace_val($0, "IS_LLM_EXTRACT", isllm) }
          if (workers != "") { $0 = replace_val($0, "LLM_WORKERS", workers) }
        }
        print
      }
    ' "$cm_file" > "$tmp_file"
    mv "$tmp_file" "$cm_file"
  else
    echo "Warning: $cm_file not found, skip updating config"
  fi

  # Apply all yaml files except workflow.yaml first
  # Then apply workflow.yaml at last
  echo "Applying manifests in $dir"
  # shellcheck disable=SC2046
  for f in $(ls "$dir"/*.yaml 2>/dev/null | grep -v "/workflow.yaml$"); do
    echo "kubectl apply -f $f"
    kubectl apply -f "$f"
  done

  if [ -f "$dir/workflow.yaml" ]; then
    WF="$dir/workflow.yaml"
    WF_TO_APPLY="$WF"
    TMP_FILE=""

    # Map directory to corresponding repo URL variable
    REPO_TO_INJECT=""
    case "$(basename "$dir")" in
      bg-deploy-civil-case)
        REPO_TO_INJECT="$GIT_REPO_CIVIL_CASE"
        ;;
      bg-deploy-criminal-case)
        REPO_TO_INJECT="$GIT_REPO_CRIMINAL_CASE"
        ;;
      bg-deploy-law)
        REPO_TO_INJECT="$GIT_REPO_CN_LAW"
        ;;
    esac

    # If a repo URL and/or gpu resource is provided, generate a temp workflow manifest injecting parameters
    if [ -n "$REPO_TO_INJECT" ] || [ -n "$GPU_RESOURCE" ]; then
      TMP_FILE="$(mktemp)"
      awk -v repo="$REPO_TO_INJECT" -v gpu="$GPU_RESOURCE" '
        BEGIN { added_repo=0; added_gpu=0 }
        # Replace existing repository-url value if present
        /^\s*-\s*name:\s*repository-url\s*$/ {
          print; getline;
          if ($0 ~ /^\s*value:\s*/) {
            sub(/^\s*value:\s*.*/, "      value: " repo);
            print; next
          } else {
            print "      value: " repo; print; next
          }
        }
        # Replace existing gpu-resource value if present
        /^\s*-\s*name:\s*gpu-resource\s*$/ {
          print; getline;
          if (length(gpu) > 0) {
            if ($0 ~ /^\s*value:\s*/) {
              sub(/^\s*value:\s*.*/, "      value: " gpu);
              print; next
            } else {
              print "      value: " gpu; print; next
            }
          }
        }
        # After parameters: line, if not yet added and not replacing, insert the repository-url item
        /^\s*parameters:\s*$/ {
          print;
          if (length(repo) > 0 && !added_repo) { print "    - name: repository-url"; print "      value: " repo; added_repo=1 }
          if (length(gpu) > 0 && !added_gpu) { print "    - name: gpu-resource"; print "      value: " gpu; added_gpu=1 }
          next
        }
        { print }
      ' "$WF" > "$TMP_FILE"
      WF_TO_APPLY="$TMP_FILE"
    fi

    if grep -q '^[[:space:]]*generateName:' "$WF_TO_APPLY"; then
      echo "kubectl create -f $WF_TO_APPLY"
      kubectl create -f "$WF_TO_APPLY"
    else
      echo "kubectl apply -f $WF_TO_APPLY"
      kubectl apply -f "$WF_TO_APPLY"
    fi

    # Clean up temp file if used
    if [ -n "$TMP_FILE" ] && [ -f "$TMP_FILE" ]; then
      rm -f "$TMP_FILE"
    fi
  fi
done

echo "All done."
