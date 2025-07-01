#!/bin/sh

# This script processes the modified files and syncs them with Dify
# It handles both file updates and deletions
# Required environment variables:
# - HOST: Dify API host
# - API_KEY: Dify API key
# - DATASET_ID: Target dataset ID
# - ALWAY_PUSH_ALL_FILES: Whether to process all files instead of only modified and created files (default: false)

set -ex

# Process deletions using the files_to_delete.txt file
echo "Processing deletions..."
if [ -f /workspace/files_to_delete.txt ]; then
  cat /workspace/files_to_delete.txt | while read -r filename; do
    if [ ! -z "$filename" ]; then
      # Find document ID for the deleted file
      doc_id=$(cat /workspace/dify_docs.json | jq -r --arg name "$filename" '.data[] | select(.name == $name) | .id')
      
      if [ ! -z "$doc_id" ]; then
        echo "File $filename was deleted, deleting document $doc_id from Dify"
        curl -v --location --request DELETE "http://${HOST}/v1/datasets/${DATASET_ID}/documents/${doc_id}" \
          --header "Authorization: Bearer ${API_KEY}"
      fi
    fi
  done
else
  echo "No deletion list found, skipping deletion processing"
fi

# Determine which files to process based on ALWAY_PUSH_ALL_FILES setting
if [ "${ALWAY_PUSH_ALL_FILES}" = "true" ]; then
  echo "Processing all files..."
  files_to_process="${WORKSPACE_DIR}/s3_files.txt.new"
else
  echo "Processing only created and modified files..."
  # Create a temporary file containing both created and modified files
  cat /workspace/files_to_create.txt /workspace/files_to_modify.txt > /workspace/files_to_process.txt
  files_to_process="/workspace/files_to_process.txt"
fi

# Process each file for creation or modification
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
    doc_id=$(cat /workspace/dify_docs.json | jq -r --arg name "$filename" '.data[] | select(.name == $name) | .id')

    # Document doesn't exist, create new one
    if [ -z "$doc_id" ]; then
      echo "Creating new document for $filename"
      curl -v --location --request POST "http://${HOST}/v1/datasets/${DATASET_ID}/document/create-by-file" \
        --header "Authorization: Bearer ${API_KEY}" \
        --form 'data={"indexing_technique":"high_quality","doc_form":"text_model","process_rule":{"rules":{"pre_processing_rules":[{"id":"remove_extra_spaces","enabled":true},{"id":"remove_urls_emails","enabled":true}],"segmentation":{"separator":"\n\n","max_tokens":512}},"mode":"custom"},"retrieval_model":{"search_method":"semantic_search","reranking_enable":true,"reranking_model":{"reranking_provider_name":"openai_api_compatible","reranking_model_name":"bge-reranker-v2-m3"},"top_k":4,"score_threshold_enabled":true,"score_threshold":0.2},"embedding_model_provider":"openai_api_compatible","embedding_model":"bge-m3"};type=text/plain' \
        --form file=@"$local_path"
    else
      echo "Updating existing document $doc_id for $filename"
      curl -v --location --request POST "http://${HOST}/v1/datasets/${DATASET_ID}/documents/${doc_id}/update-by-file" \
        --header "Authorization: Bearer ${API_KEY}" \
        --form 'data={"process_rule":{"rules":{"pre_processing_rules":[{"id":"remove_extra_spaces","enabled":true},{"id":"remove_urls_emails","enabled":true}],"segmentation":{"separator":"\n\n","max_tokens":512}},"mode":"custom"}};type=text/plain' \
        --form file=@"$local_path"
    fi
  fi
done < "$files_to_process"

# Clean up temporary file if created
if [ "${ALWAY_PUSH_ALL_FILES}" != "true" ] && [ -f /workspace/files_to_process.txt ]; then
  rm /workspace/files_to_process.txt
fi 

# Add or update metadata for version tracking
echo "Adding or updating version metadata..."

# Generate version string with current date and time
VERSION_STRING="version_$(date +'%Y%m%d_%H%M%S')"
echo "Version string: $VERSION_STRING"

# First, check if version metadata already exists
METADATA_RESPONSE=$(curl --silent --location --request GET "http://${HOST}/v1/datasets/${DATASET_ID}/metadata" \
  --header "Authorization: Bearer ${API_KEY}")

# Extract version metadata ID if it exists
VERSION_METADATA_ID=$(echo "$METADATA_RESPONSE" | jq -r '.doc_metadata[] | select(.name | startswith("version_")) | .id' 2>/dev/null || echo "")

if [ -z "$VERSION_METADATA_ID" ] || [ "$VERSION_METADATA_ID" = "null" ]; then
  # Version metadata doesn't exist, create it
  echo "Creating new version metadata..."
  curl --location --request POST "http://${HOST}/v1/datasets/${DATASET_ID}/metadata" \
    --header "Authorization: Bearer ${API_KEY}" \
    --header "Content-Type: application/json" \
    --data-raw "{\"type\": \"string\", \"name\": \"$VERSION_STRING\"}"
else
  # Version metadata exists, update it
  echo "Updating existing version metadata (ID: $VERSION_METADATA_ID)..."
  curl --location --request PATCH "http://${HOST}/v1/datasets/${DATASET_ID}/metadata/${VERSION_METADATA_ID}" \
    --header "Authorization: Bearer ${API_KEY}" \
    --header "Content-Type: application/json" \
    --data-raw "{\"name\": \"$VERSION_STRING\"}"
fi

# Move new s3_files.txt into place - only done after successful completion of the entire workflow
mv /workspace/s3_files.txt.new /workspace/s3_files.txt 
