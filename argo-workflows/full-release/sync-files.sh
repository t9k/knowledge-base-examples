#!/bin/sh

# This script fetches files from S3 bucket
# It uses the following environment variables from configmap:
# - S3_PATH: S3 path pattern for matching files

set -ex

# Load environment variables
if [ -f /env-config/env.properties ]; then
  source /env-config/env.properties
fi

# Extract bucket name from S3_PATH
S3_BUCKET=$(echo "$S3_PATH" | cut -d'/' -f3)
S3_PREFIX=$(echo "$S3_PATH" | cut -d'/' -f4-)

# Copy rclone config
mkdir -p /root/.config/rclone
cp /s3cfg/rclone.conf /root/.config/rclone/rclone.conf

# Create workspace directory
mkdir -p /workspace/files

# Sync only .txt and .md files
echo "Syncing .txt and .md files from S3 path ${S3_PATH}..."
rclone sync --config /root/.config/rclone/rclone.conf --include "*.txt" --include "*.md" minio:${S3_BUCKET}/${S3_PREFIX} /workspace/files/

# Create s3_files.txt with file paths and modification date/time
echo "Creating list of all files with modification times..."
> /workspace/s3_files.txt
rclone lsl --config /root/.config/rclone/rclone.conf --include "*.txt" --include "*.md" minio:${S3_BUCKET}/${S3_PREFIX} | while read -r line; do
  file=$(echo "$line" | awk '{print $4}')
  mod_date=$(echo "$line" | awk '{print $2}')
  mod_time=$(echo "$line" | awk '{print $3}')
  echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_files.txt
done

# List synced files
echo "Synced files:"
ls -la /workspace/files/

# Print stats
echo "Files synced: $(wc -l < /workspace/s3_files.txt)"
