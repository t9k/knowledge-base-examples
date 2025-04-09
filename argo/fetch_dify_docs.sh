#!/bin/sh

# This script fetches the list of existing documents from Dify API
# It requires the following environment variables:
# - HOST: Dify API host
# - API_KEY: Dify API key
# - DATASET_ID: Target dataset ID

set -ex

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
echo "$response" > /workspace/dify_docs.json
