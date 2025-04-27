#!/bin/sh

# This script fetches files from S3 bucket
# It uses the following environment variables from configmap:
# - S3_PATH_TEXT: S3 path pattern for matching text files
# - S3_PATH_IMAGE: S3 path pattern for matching image files

set -e

# Load environment variables
if [ -f /env-config/env.properties ]; then
  source /env-config/env.properties
fi

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting sync-files.sh script"

# Copy rclone config
mkdir -p /root/.config/rclone
cp /s3cfg/rclone.conf /root/.config/rclone/rclone.conf
echo "$(date +'%Y-%m-%d %H:%M:%S') - Rclone config copied"

# Create workspace directory
mkdir -p /workspace/files
echo "$(date +'%Y-%m-%d %H:%M:%S') - Workspace directories created"

# Function to process text files
process_text_files() {
  # Extract bucket name and prefix from S3_PATH_TEXT
  TEXT_S3_BUCKET=$(echo "$S3_PATH_TEXT" | cut -d'/' -f3)
  TEXT_S3_PREFIX=$(echo "$S3_PATH_TEXT" | cut -d'/' -f4- | sed 's/\/$//')
  
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Text S3 bucket: ${TEXT_S3_BUCKET}"
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Text S3 prefix: ${TEXT_S3_PREFIX}"
  
  # Create directory name based on S3 path
  TEXT_DIR_NAME=$(echo "${TEXT_S3_BUCKET}--${TEXT_S3_PREFIX}" | tr '/' '--')
  
  # Create directory for text files
  mkdir -p "/workspace/files/${TEXT_DIR_NAME}"
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Created directory: /workspace/files/${TEXT_DIR_NAME}"
  
  # Sync text files (.txt and .md)
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Syncing text files (.txt and .md) from S3 path ${S3_PATH_TEXT}..."
  rclone sync --config /root/.config/rclone/rclone.conf \
    --include "*.txt" --include "*.md" \
    minio:${TEXT_S3_BUCKET}/${TEXT_S3_PREFIX} \
    /workspace/files/${TEXT_DIR_NAME}/
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Text files sync completed"
  
  # Create text file list with file paths and modification date/time
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Creating list of text files with modification times..."
  > /workspace/s3_text_file_list.txt
  rclone lsl --config /root/.config/rclone/rclone.conf \
    --include "*.txt" --include "*.md" \
    minio:${TEXT_S3_BUCKET}/${TEXT_S3_PREFIX} | while read -r line; do
    file=$(echo "$line" | awk '{print $4}')
    mod_date=$(echo "$line" | awk '{print $2}')
    mod_time=$(echo "$line" | awk '{print $3}')
    echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_text_file_list.txt
  done
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Text file list created at /workspace/s3_text_file_list.txt"
  
  # Print stats for text files
  FILE_COUNT=$(wc -l < /workspace/s3_text_file_list.txt)
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Text files synced: ${FILE_COUNT}"
}

# Function to process image files
process_image_files() {
  # Extract bucket name and prefix from S3_PATH_IMAGE
  IMAGE_S3_BUCKET=$(echo "$S3_PATH_IMAGE" | cut -d'/' -f3)
  IMAGE_S3_PREFIX=$(echo "$S3_PATH_IMAGE" | cut -d'/' -f4- | sed 's/\/$//')
  
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Image S3 bucket: ${IMAGE_S3_BUCKET}"
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Image S3 prefix: ${IMAGE_S3_PREFIX}"
  
  # Create directory name based on S3 path
  IMAGE_DIR_NAME=$(echo "${IMAGE_S3_BUCKET}--${IMAGE_S3_PREFIX}" | tr '/' '--')
  
  # Create directory for image files
  mkdir -p "/workspace/files/${IMAGE_DIR_NAME}"
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Created directory: /workspace/files/${IMAGE_DIR_NAME}"
  
  # Sync image files (.jpg, .jpeg, .png)
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Syncing image files (.jpg, .jpeg, .png) from S3 path ${S3_PATH_IMAGE}..."
  rclone sync --config /root/.config/rclone/rclone.conf \
    --include "*.jpg" --include "*.jpeg" --include "*.png" \
    minio:${IMAGE_S3_BUCKET}/${IMAGE_S3_PREFIX} \
    /workspace/files/${IMAGE_DIR_NAME}/
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Image files sync completed"
  
  # Create image file list with file paths and modification date/time
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Creating list of image files with modification times..."
  > /workspace/s3_image_file_list.txt
  rclone lsl --config /root/.config/rclone/rclone.conf \
    --include "*.jpg" --include "*.jpeg" --include "*.png" \
    minio:${IMAGE_S3_BUCKET}/${IMAGE_S3_PREFIX} | while read -r line; do
    file=$(echo "$line" | awk '{print $4}')
    mod_date=$(echo "$line" | awk '{print $2}')
    mod_time=$(echo "$line" | awk '{print $3}')
    echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_image_file_list.txt
  done
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Image file list created at /workspace/s3_image_file_list.txt"
  
  # Print stats for image files
  FILE_COUNT=$(wc -l < /workspace/s3_image_file_list.txt)
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Image files synced: ${FILE_COUNT}"
}

# Process text files if S3_PATH_TEXT is provided
if [ -n "$S3_PATH_TEXT" ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting text file processing"
  process_text_files
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Text file processing completed"
else
  echo "$(date +'%Y-%m-%d %H:%M:%S') - S3_PATH_TEXT is not set. Skipping text file processing."
fi

# Process image files if S3_PATH_IMAGE is provided
if [ -n "$S3_PATH_IMAGE" ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting image file processing"
  process_image_files
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Image file processing completed"
else
  echo "$(date +'%Y-%m-%d %H:%M:%S') - S3_PATH_IMAGE is not set. Skipping image file processing."
fi

# List all synced files
echo "$(date +'%Y-%m-%d %H:%M:%S') - All synced files:"
find /workspace/files -type f | sort

echo "$(date +'%Y-%m-%d %H:%M:%S') - sync-files.sh completed successfully"
