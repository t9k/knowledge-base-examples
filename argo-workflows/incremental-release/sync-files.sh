#!/bin/sh

# This script fetches files from S3 bucket and tracks changes for incremental deployment
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

# Create log file if it doesn't exist, otherwise append to it
LOG_FILE="/workspace/log.txt"
if [ ! -f $LOG_FILE ]; then
  echo -e "\n\n====================SYNC_BOUNDARY====================" > $LOG_FILE
  echo "--- Sync Files $(date) ---" >> $LOG_FILE
else
  echo -e "\n\n====================SYNC_BOUNDARY====================" >> $LOG_FILE
  echo "--- Sync Files $(date) ---" >> $LOG_FILE
fi

# Create s3_files.txt if it doesn't exist (first run case)
if [ ! -f /workspace/s3_files.txt ]; then
  echo "First run detected, creating empty s3_files.txt file" | tee -a $LOG_FILE
  touch /workspace/s3_files.txt
fi

# Sync only .txt and .md files
echo "Syncing .txt and .md files from S3 path ${S3_PATH}..." | tee -a $LOG_FILE
rclone sync --config /root/.config/rclone/rclone.conf --include "*.txt" --include "*.md" minio:${S3_BUCKET}/${S3_PREFIX} /workspace/files/

# Create s3_files.txt.new with file paths and modification date/time
echo "Creating list of all files with modification times..." | tee -a $LOG_FILE
> /workspace/s3_files.txt.new
rclone lsl --config /root/.config/rclone/rclone.conf --include "*.txt" --include "*.md" minio:${S3_BUCKET}/${S3_PREFIX} | while read -r line; do
  file=$(echo "$line" | awk '{print $4}')
  mod_date=$(echo "$line" | awk '{print $2}')
  mod_time=$(echo "$line" | awk '{print $3}')
  echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_files.txt.new
done

# Identify changes between current and new file lists
echo "Identifying file changes for incremental update..." | tee -a $LOG_FILE

# Find created and modified files
while read -r line; do
  file=$(echo "$line" | awk '{print $1}')
  old_file_exists=$(grep -q "^${file} " /workspace/s3_files.txt && echo "yes" || echo "no")
  
  if [ "$old_file_exists" = "no" ]; then
    # File is newly created
    echo "TO_CREATE: $file" | tee -a $LOG_FILE
  else
    # Check if file was modified
    old_mod_date=$(grep "^${file} " /workspace/s3_files.txt | awk '{print $2}')
    old_mod_time=$(grep "^${file} " /workspace/s3_files.txt | awk '{print $3}')
    new_mod_date=$(grep "^${file} " /workspace/s3_files.txt.new | awk '{print $2}')
    new_mod_time=$(grep "^${file} " /workspace/s3_files.txt.new | awk '{print $3}')
    
    if [ "$old_mod_date" != "$new_mod_date" ] || [ "$old_mod_time" != "$new_mod_time" ]; then
      echo "TO_MODIFY: $file" | tee -a $LOG_FILE
    fi
  fi
done < /workspace/s3_files.txt.new

# Find deleted files - read from old file list and check against new file list
while read -r line; do
  file=$(echo "$line" | awk '{print $1}')
  new_file_exists=$(grep -q "^${file} " /workspace/s3_files.txt.new && echo "yes" || echo "no")
  
  if [ "$new_file_exists" = "no" ]; then
    # File was deleted
    echo "TO_DELETE: $file" | tee -a $LOG_FILE
  fi
done < /workspace/s3_files.txt

# List synced files
echo "Synced files:" | tee -a $LOG_FILE
ls -l /workspace/files/ | tee -a $LOG_FILE

# Print stats
echo "Files synced: $(wc -l < /workspace/s3_files.txt.new)" | tee -a $LOG_FILE
echo "Changes complete. New files list created but not applied yet." | tee -a $LOG_FILE
