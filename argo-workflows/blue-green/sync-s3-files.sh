#!/bin/sh

# This script fetches files from S3 bucket
# It requires the following environment variables:
# - ALWAY_PUSH_ALL_FILES: Whether to process all files or only created/modified files (default: false)
# - WORKSPACE_DIR: The active environment workspace directory (/workspace/blue or /workspace/green)

set -ex

# Default workspace directory if not set
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Copy rclone config
mkdir -p /root/.config/rclone
cp /s3cfg/rclone.conf /root/.config/rclone/rclone.conf

# Make sure shared files directory exists
mkdir -p /workspace/files

# Function to get file modification date and time
get_mod_datetime() {
  rclone lsl --config /root/.config/rclone/rclone.conf minio:${S3_BUCKET}/${1} | awk '{print $2 " " $3}'
}

# Always sync all files to shared files directory
echo "Syncing all files from S3..."
rclone sync --config /root/.config/rclone/rclone.conf minio:${S3_BUCKET} /workspace/files/

# Create s3_files.txt with file paths and modification date/time
echo "Creating list of all files with modification times..."
rclone lsl --config /root/.config/rclone/rclone.conf minio:${S3_BUCKET} | while read -r line; do
  file=$(echo "$line" | awk '{print $4}')
  mod_date=$(echo "$line" | awk '{print $2}')
  mod_time=$(echo "$line" | awk '{print $3}')
  echo "${file} ${mod_date} ${mod_time}" >> ${WORKSPACE_DIR}/s3_files.txt.new
done

# Create or clear files_to_modify.txt, files_to_create.txt, and files_to_delete.txt
> ${WORKSPACE_DIR}/files_to_modify.txt
> ${WORKSPACE_DIR}/files_to_create.txt
> ${WORKSPACE_DIR}/files_to_delete.txt

# Cold start scenario: If s3_files.txt doesn't exist, treat all files as created
if [ ! -f ${WORKSPACE_DIR}/s3_files.txt ]; then
  echo "Cold start detected (no previous s3_files.txt). Treating all files as created..."
  # Extract filenames from s3_files.txt.new and add to files_to_create.txt
  cat ${WORKSPACE_DIR}/s3_files.txt.new | awk '{print $1}' >> ${WORKSPACE_DIR}/files_to_create.txt
  echo "Added $(wc -l < ${WORKSPACE_DIR}/files_to_create.txt) files to files_to_create.txt"
# Normal case: If s3_files.txt exists and is not empty, compare and create files_to_modify.txt, files_to_create.txt, and files_to_delete.txt
elif [ -s ${WORKSPACE_DIR}/s3_files.txt ]; then
  echo "Checking for modified, created, and deleted files..."
  
  # Find created and modified files
  while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    old_file_exists=$(grep -q "^${file} " ${WORKSPACE_DIR}/s3_files.txt && echo "yes" || echo "no")
    
    if [ "$old_file_exists" = "no" ]; then
      # File is newly created
      echo "$file" >> ${WORKSPACE_DIR}/files_to_create.txt
    else
      # Check if file was modified
      old_mod_date=$(grep "^${file} " ${WORKSPACE_DIR}/s3_files.txt | awk '{print $2}')
      old_mod_time=$(grep "^${file} " ${WORKSPACE_DIR}/s3_files.txt | awk '{print $3}')
      new_mod_datetime=$(get_mod_datetime "$file")
      new_mod_date=$(echo "$new_mod_datetime" | awk '{print $1}')
      new_mod_time=$(echo "$new_mod_datetime" | awk '{print $2}')
      
      if [ "$old_mod_date" != "$new_mod_date" ] || [ "$old_mod_time" != "$new_mod_time" ]; then
        echo "$file" >> ${WORKSPACE_DIR}/files_to_modify.txt
      fi
    fi
  done < ${WORKSPACE_DIR}/s3_files.txt.new
  
  # Find deleted files - read from old file list and check against new file list
  while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    new_file_exists=$(grep -q "^${file} " ${WORKSPACE_DIR}/s3_files.txt.new && echo "yes" || echo "no")
    
    if [ "$new_file_exists" = "no" ]; then
      # File was deleted
      echo "$file" >> ${WORKSPACE_DIR}/files_to_delete.txt
    fi
  done < ${WORKSPACE_DIR}/s3_files.txt
else
  # s3_files.txt exists but is empty
  echo "Previous s3_files.txt is empty. Treating all files as created..."
  # Extract filenames from s3_files.txt.new and add to files_to_create.txt
  cat ${WORKSPACE_DIR}/s3_files.txt.new | awk '{print $1}' >> ${WORKSPACE_DIR}/files_to_create.txt
  echo "Added $(wc -l < ${WORKSPACE_DIR}/files_to_create.txt) files to files_to_create.txt"
fi

# List synced files
echo "Synced files:"
ls -la /workspace/files/

# Print stats
echo "Files to create: $(wc -l < ${WORKSPACE_DIR}/files_to_create.txt)"
echo "Files to modify: $(wc -l < ${WORKSPACE_DIR}/files_to_modify.txt)"
echo "Files to delete: $(wc -l < ${WORKSPACE_DIR}/files_to_delete.txt)"
