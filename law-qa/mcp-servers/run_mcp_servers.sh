#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 --milvus-uri=<URI> --embedding-base-url=<URL> [--milvus-db=<DB>] [--reranker-base-url=<URL>] [--enable-auth=<true|false>] [--gpu-vendor=<vendor>]"
}

MILVUS_URI=""
EMBEDDING_BASE_URL=""
RERANKER_BASE_URL=""
ENABLE_AUTH=""
GPU_VENDOR=""
MILVUS_DB="default"

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
    --reranker-base-url=*)
      RERANKER_BASE_URL="${1#*=}"
      ;;
    --reranker-base-url)
      shift
      RERANKER_BASE_URL="${1:-}"
      ;;
    --enable-auth=*)
      ENABLE_AUTH="${1#*=}"
      ;;
    --enable-auth)
      shift
      ENABLE_AUTH="${1:-}"
      ;;
    --milvus-db=*)
      MILVUS_DB="${1#*=}"
      ;;
    --milvus-db)
      shift
      MILVUS_DB="${1:-}"
      ;;
    --gpu-vendor=*)
      GPU_VENDOR="${1#*=}"
      ;;
    --gpu-vendor)
      shift
      GPU_VENDOR="${1:-}"
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

if [ -z "$MILVUS_URI" ] || [ -z "$EMBEDDING_BASE_URL" ]; then
  echo "Error: --milvus-uri and --embedding-base-url are required." >&2
  usage
  exit 1
fi

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

update_yaml_file() {
  local file="$1"
  local tmp_file="$file.tmp"

  # Escape values for sed replacement (preserve & and |)
  escape_sed() {
    printf '%s' "$1" | sed -e 's/[&|]/\\&/g'
  }

  local sed_cmds=()
  if [ -n "$MILVUS_URI" ]; then
    local v
    v="$(escape_sed "$MILVUS_URI")"
    sed_cmds+=("-e" "s|^([[:space:]]*)MILVUS_URI:[[:space:]]*.*$|\\1MILVUS_URI: \"$v\"|")
  fi
  if [ -n "$EMBEDDING_BASE_URL" ]; then
    local v
    v="$(escape_sed "$EMBEDDING_BASE_URL")"
    sed_cmds+=("-e" "s|^([[:space:]]*)EMBEDDING_BASE_URL:[[:space:]]*.*$|\\1EMBEDDING_BASE_URL: \"$v\"|")
  fi
  if [ -n "$RERANKER_BASE_URL" ]; then
    local v
    v="$(escape_sed "$RERANKER_BASE_URL")"
    sed_cmds+=("-e" "s|^([[:space:]]*)RERANKER_BASE_URL:[[:space:]]*.*$|\\1RERANKER_BASE_URL: \"$v\"|")
  fi
  if [ -n "$ENABLE_AUTH" ]; then
    local v
    v="$(escape_sed "$ENABLE_AUTH")"
    sed_cmds+=("-e" "s|^([[:space:]]*)ENABLE_AUTH:[[:space:]]*.*$|\\1ENABLE_AUTH: \"$v\"|")
  fi
  if [ -n "$MILVUS_DB" ]; then
    local v
    v="$(escape_sed "$MILVUS_DB")"
    sed_cmds+=("-e" "s|^([[:space:]]*)MILVUS_DB:[[:space:]]*.*$|\\1MILVUS_DB: \"$v\"|")
  fi

  # When GPU vendor is enflame, swap the two image lines (uncomment qy, comment cn)
  if [ "${GPU_VENDOR:-}" = "enflame" ]; then
    # Uncomment the qy image line
    sed_cmds+=("-e" "s|^([[:space:]]*)# image:[[:space:]]*registry\\.qy\\.t9kcloud\\.cn/topsrider/mcp-server:case-searcher-20250904[[:space:]]*$|\\1image: registry.qy.t9kcloud.cn/topsrider/mcp-server:case-searcher-20250904|")
    # Comment the cn-hangzhou image line
    sed_cmds+=("-e" "s|^([[:space:]]*)image:[[:space:]]*registry\\.cn-hangzhou\\.aliyuncs\\.com/t9k/mcp-server:case-searcher-20250904[[:space:]]*$|\\1# image: registry.cn-hangzhou.aliyuncs.com/t9k/mcp-server:case-searcher-20250904|")

    # For law-searcher image lines as well
    sed_cmds+=("-e" "s|^([[:space:]]*)# image:[[:space:]]*registry\\.qy\\.t9kcloud\\.cn/topsrider/mcp-server:law-searcher-20250904[[:space:]]*$|\\1image: registry.qy.t9kcloud.cn/topsrider/mcp-server:law-searcher-20250904|")
    sed_cmds+=("-e" "s|^([[:space:]]*)image:[[:space:]]*registry\\.cn-hangzhou\\.aliyuncs\\.com/t9k/mcp-server:law-searcher-20250904[[:space:]]*$|\\1# image: registry.cn-hangzhou.aliyuncs.com/t9k/mcp-server:law-searcher-20250904|")

    # Swap GPU resource lines to enflame
    sed_cmds+=("-e" "s|^([[:space:]]*)# [[:space:]]*enflame\\.com/gcu:[[:space:]]*.*$|\\1enflame.com/gcu: 1|")
    sed_cmds+=("-e" "s|^([[:space:]]*)(nvidia\\.com/gpu:[[:space:]]*[^#].*)$|\\1# \\2|")
  fi

  # If GPU vendor is None, comment out all NVIDIA GPU requests/limits
  if [ "${GPU_VENDOR:-}" = "None" ] || [ "${GPU_VENDOR:-}" = "none" ] || [ "${GPU_VENDOR:-}" = "NONE" ]; then
    sed_cmds+=("-e" "s|^([[:space:]]*)(nvidia\\.com/gpu:[[:space:]]*[^#].*)$|\\1# \\2|")
  fi

  if [ ${#sed_cmds[@]} -gt 0 ]; then
    sed -E "${sed_cmds[@]}" "$file" > "$tmp_file" && mv "$tmp_file" "$file"
  fi
}

# Process all service directories under mcp-servers
for dir in "$BASE_DIR"/*; do
  [ -d "$dir" ] || continue
  f="$dir/k8s.yaml"
  if [ -f "$f" ]; then
    echo "Processing directory: $dir"
    echo "Updating $f"
    update_yaml_file "$f"
    echo "kubectl apply -f $f"
    kubectl apply -f "$f"
  fi
done

echo "All MCP server manifests applied."
