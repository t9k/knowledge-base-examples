#!/bin/sh

# This script fetches files from S3 bucket
# It requires the following environment variables:
# - MODIFIED_ONLY: Whether to only process files listed in modified_files.txt (default: true)

set -ex

# Copy rclone config
mkdir -p /root/.config/rclone
cp /s3cfg/rclone.conf /root/.config/rclone/rclone.conf

# Create workspace directory
mkdir -p /workspace/files

# Function to get file modification date and time
get_mod_datetime() {
  rclone lsl --config /root/.config/rclone/rclone.conf minio:${S3_BUCKET}/${1} | awk '{print $2 " " $3}'
}

# Always sync all files
echo "Syncing all files from S3..."
rclone sync --config /root/.config/rclone/rclone.conf minio:${S3_BUCKET} /workspace/files/

# Create s3_files.txt with file paths and modification date/time
echo "Creating list of all files with modification times..."
rclone lsl --config /root/.config/rclone/rclone.conf minio:${S3_BUCKET} | while read -r line; do
  file=$(echo "$line" | awk '{print $4}')
  mod_date=$(echo "$line" | awk '{print $2}')
  mod_time=$(echo "$line" | awk '{print $3}')
  echo "${file} ${mod_date} ${mod_time}" >> /workspace/s3_files.txt.new
done

# Create or clear modified_files.txt
> /workspace/modified_files.txt

# If s3_files.txt exists, compare and create modified_files.txt
if [ -f /workspace/s3_files.txt ]; then
  echo "Checking for modified files..."
  while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    old_mod_date=$(echo "$line" | awk '{print $2}')
    old_mod_time=$(echo "$line" | awk '{print $3}')
    new_mod_datetime=$(get_mod_datetime "$file")
    new_mod_date=$(echo "$new_mod_datetime" | awk '{print $1}')
    new_mod_time=$(echo "$new_mod_datetime" | awk '{print $2}')
    
    if [ "$old_mod_date" != "$new_mod_date" ] || [ "$old_mod_time" != "$new_mod_time" ]; then
      echo "$file" >> /workspace/modified_files.txt
    fi
  done < /workspace/s3_files.txt
fi

# Move new s3_files.txt into place
mv /workspace/s3_files.txt.new /workspace/s3_files.txt

# List synced files
echo "Synced files:"
ls -la /workspace/files/
