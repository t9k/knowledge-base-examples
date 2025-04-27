#!/bin/sh

# This script fetches files from S3 bucket
# It uses the following environment variables from configmap:
# - S3_PATH_TEXT: S3 path pattern for matching text files
# - S3_PATH_IMAGE: S3 path pattern for matching image files

set -ex

# Load environment variables
if [ -f /env-config/env.properties ]; then
  source /env-config/env.properties
fi

# Copy rclone config
mkdir -p /root/.config/rclone
cp /s3cfg/rclone.conf /root/.config/rclone/rclone.conf

# Create workspace directory
mkdir -p /workspace/files

# Function to process text files
process_text_files() {
  # Extract bucket name and prefix from S3_PATH_TEXT
  TEXT_S3_BUCKET=$(echo "$S3_PATH_TEXT" | cut -d'/' -f3)
  TEXT_S3_PREFIX=$(echo "$S3_PATH_TEXT" | cut -d'/' -f4- | sed 's/\/$//')
  
  # Create directory name based on S3 path
  TEXT_DIR_NAME=$(echo "${TEXT_S3_BUCKET}--${TEXT_S3_PREFIX}" | tr '/' '--')
  
  # Create directory for text files
  mkdir -p "/workspace/files/${TEXT_DIR_NAME}"
  
  # Sync text files (.txt and .md)
  echo "Syncing text files (.txt and .md) from S3 path ${S3_PATH_TEXT}..."
  rclone sync --config /root/.config/rclone/rclone.conf \
    --include "*.txt" --include "*.md" \
    minio:${TEXT_S3_BUCKET}/${TEXT_S3_PREFIX} \
    /workspace/files/${TEXT_DIR_NAME}/
  
  # Create text file list with file paths and modification date/time
  echo "Creating list of text files with modification times..."
  > /workspace/s3_text_file_list.txt
  rclone lsl --config /root/.config/rclone/rclone.conf \
    --include "*.txt" --include "*.md" \
    minio:${TEXT_S3_BUCKET}/${TEXT_S3_PREFIX} | while read -r line; do
    file=$(echo "$line" | awk '{print $4}')
    mod_date=$(echo "$line" | awk '{print $2}')
    mod_time=$(echo "$line" | awk '{print $3}')
    echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_text_file_list.txt
  done
  
  # Print stats for text files
  echo "Text files synced: $(wc -l < /workspace/s3_text_file_list.txt)"
}

# Function to process image files
process_image_files() {
  # Extract bucket name and prefix from S3_PATH_IMAGE
  IMAGE_S3_BUCKET=$(echo "$S3_PATH_IMAGE" | cut -d'/' -f3)
  IMAGE_S3_PREFIX=$(echo "$S3_PATH_IMAGE" | cut -d'/' -f4- | sed 's/\/$//')
  
  # Create directory name based on S3 path
  IMAGE_DIR_NAME=$(echo "${IMAGE_S3_BUCKET}--${IMAGE_S3_PREFIX}" | tr '/' '--')
  
  # Create directory for image files
  mkdir -p "/workspace/files/${IMAGE_DIR_NAME}"
  
  # Sync image files (.jpg, .jpeg, .png)
  echo "Syncing image files (.jpg, .jpeg, .png) from S3 path ${S3_PATH_IMAGE}..."
  rclone sync --config /root/.config/rclone/rclone.conf \
    --include "*.jpg" --include "*.jpeg" --include "*.png" \
    minio:${IMAGE_S3_BUCKET}/${IMAGE_S3_PREFIX} \
    /workspace/files/${IMAGE_DIR_NAME}/
  
  # Create image file list with file paths and modification date/time
  echo "Creating list of image files with modification times..."
  > /workspace/s3_image_file_list.txt
  rclone lsl --config /root/.config/rclone/rclone.conf \
    --include "*.jpg" --include "*.jpeg" --include "*.png" \
    minio:${IMAGE_S3_BUCKET}/${IMAGE_S3_PREFIX} | while read -r line; do
    file=$(echo "$line" | awk '{print $4}')
    mod_date=$(echo "$line" | awk '{print $2}')
    mod_time=$(echo "$line" | awk '{print $3}')
    echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_image_file_list.txt
  done
  
  # Print stats for image files
  echo "Image files synced: $(wc -l < /workspace/s3_image_file_list.txt)"
}

# Process text files if S3_PATH_TEXT is provided
if [ -n "$S3_PATH_TEXT" ]; then
  process_text_files
else
  echo "S3_PATH_TEXT is not set. Skipping text file processing."
fi

# Process image files if S3_PATH_IMAGE is provided
if [ -n "$S3_PATH_IMAGE" ]; then
  process_image_files
else
  echo "S3_PATH_IMAGE is not set. Skipping image file processing."
fi

# List all synced files
echo "All synced files:"
find /workspace/files -type f | sort
