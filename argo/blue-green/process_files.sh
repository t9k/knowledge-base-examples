#!/bin/sh

# This script processes the modified files and syncs them with Dify
# It handles both file updates and deletions
# Required environment variables:
# - HOST: Dify API host
# - API_KEY: Dify API key
# - DATASET_ID: Target dataset ID
# - MODIFIED_ONLY: Whether to only process files listed in modified_files.txt (default: true)
# - WORKSPACE_DIR: The active environment workspace directory (/workspace/blue or /workspace/green)

set -ex

# Default workspace directory if not set
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Process deletions - check each document in Dify against all files
echo "Processing deletions..."
cat ${WORKSPACE_DIR}/dify_docs.json | jq -r '.data[] | .name + " " + .id' | while read -r doc_info; do
  doc_name=$(echo "$doc_info" | cut -d' ' -f1)
  doc_id=$(echo "$doc_info" | cut -d' ' -f2)
  
  # Check if file exists in workspace
  if ! grep -q "^${doc_name} " ${WORKSPACE_DIR}/s3_files.txt; then
    echo "File $doc_name no longer exists in workspace, deleting document $doc_id"
    curl -v --location --request DELETE "http://${HOST}/v1/datasets/${DATASET_ID}/documents/${doc_id}" \
      --header "Authorization: Bearer ${API_KEY}"
  fi
done

# Process files based on MODIFIED_ONLY setting
if [ "${MODIFIED_ONLY}" = "true" ] && [ -f ${WORKSPACE_DIR}/modified_files.txt ]; then
  echo "Processing only modified files..."
  files_to_process="${WORKSPACE_DIR}/modified_files.txt"
else
  echo "Processing all files..."
  files_to_process="${WORKSPACE_DIR}/s3_files.txt"
fi

# Process each file
while read -r line; do
  if [ ! -z "$line" ]; then
    # Extract filename from the line (first field before space)
    filename=$(echo "$line" | awk '{print $1}')
    local_path="/workspace/files/${filename}"
    
    echo "Processing file: $local_path"
    
    if [ ! -f "$local_path" ]; then
      echo "Error: File not found: $local_path"
      continue
    fi
    
    # Check if document already exists
    doc_id=$(cat ${WORKSPACE_DIR}/dify_docs.json | jq -r --arg name "$filename" '.data[] | select(.name == $name) | .id')
    
    if [ -z "$doc_id" ]; then
      # Document doesn't exist, create new one
      echo "Creating new document for $filename"
      curl -v --location --request POST "http://${HOST}/v1/datasets/${DATASET_ID}/document/create-by-file" \
        --header "Authorization: Bearer ${API_KEY}" \
        --form 'data={"indexing_technique":"high_quality","doc_form":"text_model","process_rule":{"rules":{"pre_processing_rules":[{"id":"remove_extra_spaces","enabled":true},{"id":"remove_urls_emails","enabled":true}],"segmentation":{"separator":"\n\n","max_tokens":512}},"mode":"custom"},"retrieval_model":{"search_method":"semantic_search","reranking_enable":true,"reranking_model":{"reranking_provider_name":"openai_api_compatible","reranking_model_name":"bge-reranker-v2-m3"},"top_k":4,"score_threshold_enabled":true,"score_threshold":0.2},"embedding_model_provider":"openai_api_compatible","embedding_model":"bge-m3"};type=text/plain' \
        --form file=@"$local_path"
    else
      # Document exists, update it
      echo "Updating existing document $doc_id for $filename"
      curl -v --location --request POST "http://${HOST}/v1/datasets/${DATASET_ID}/documents/${doc_id}/update-by-file" \
        --header "Authorization: Bearer ${API_KEY}" \
        --form 'data={"process_rule":{"rules":{"pre_processing_rules":[{"id":"remove_extra_spaces","enabled":true},{"id":"remove_urls_emails","enabled":true}],"segmentation":{"separator":"\n\n","max_tokens":512}},"mode":"custom"}};type=text/plain' \
        --form file=@"$local_path"
    fi
  fi
done < "$files_to_process" 
