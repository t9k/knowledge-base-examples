#!/bin/sh

# This script fetches files from S3 bucket
# It uses the following environment variables from configmap:
# - S3_PATH: S3 path pattern for matching files

set -e

# Load environment variables
if [ -f /env-config/env.properties ]; then
  source /env-config/env.properties
fi

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting sync-files.sh script"

# Extract bucket name from S3_PATH
S3_BUCKET=$(echo "$S3_PATH" | cut -d'/' -f3)
S3_PREFIX=$(echo "$S3_PATH" | cut -d'/' -f4-)

echo "S3 bucket: ${S3_BUCKET}"
echo "S3 prefix: ${S3_PREFIX}"

# Copy rclone config
mkdir -p /root/.config/rclone
cp /s3cfg/rclone.conf /root/.config/rclone/rclone.conf
echo "Rclone config copied"

# Create workspace directory
mkdir -p /workspace/files
echo "Workspace directories created"

# Sync only .txt and .md files
echo "$(date +'%Y-%m-%d %H:%M:%S') - Syncing .txt and .md files from S3 path ${S3_PATH}..."
rclone sync --config /root/.config/rclone/rclone.conf --include "*.txt" --include "*.md" minio:${S3_BUCKET}/${S3_PREFIX} /workspace/files/
echo "$(date +'%Y-%m-%d %H:%M:%S') - File sync completed"

# Create s3_file_list.txt with file paths and modification date/time
echo "$(date +'%Y-%m-%d %H:%M:%S') - Creating list of all files with modification times..."
> /workspace/s3_file_list.txt
rclone lsl --config /root/.config/rclone/rclone.conf --include "*.txt" --include "*.md" minio:${S3_BUCKET}/${S3_PREFIX} | while read -r line; do
  file=$(echo "$line" | awk '{print $4}')
  mod_date=$(echo "$line" | awk '{print $2}')
  mod_time=$(echo "$line" | awk '{print $3}')
  echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_file_list.txt
done
echo "$(date +'%Y-%m-%d %H:%M:%S') - File list created at /workspace/s3_file_list.txt"

# List synced files
echo "$(date +'%Y-%m-%d %H:%M:%S') - Synced files:"
ls -la /workspace/files/

# Print stats
FILE_COUNT=$(wc -l < /workspace/s3_file_list.txt)
echo "$(date +'%Y-%m-%d %H:%M:%S') - Files synced: ${FILE_COUNT}"
echo "$(date +'%Y-%m-%d %H:%M:%S') - sync-files.sh completed successfully"
