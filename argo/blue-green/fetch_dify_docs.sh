#!/bin/sh

# This script fetches the list of existing documents from Dify API
# It requires the following environment variables:
# - HOST: Dify API host
# - API_KEY: Dify API key
# - DATASET_ID: Target dataset ID
# - WORKSPACE_DIR: The active environment workspace directory (/workspace/blue or /workspace/green)

set -ex

# Default workspace directory if not set
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Fetch document list from Dify API
response=$(curl --location --request GET "http://${HOST}/v1/datasets/${DATASET_ID}/documents" \
  --header "Authorization: Bearer ${API_KEY}" \
  --silent)

# Validate response is valid JSON
if ! echo "$response" | jq . >/dev/null 2>&1; then
  echo "Error: Invalid JSON response from API"
  exit 1
fi

# Write to output file
echo "$response" > ${WORKSPACE_DIR}/dify_docs.json
