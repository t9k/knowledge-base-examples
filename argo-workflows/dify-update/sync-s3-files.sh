#!/bin/sh

# This script fetches files from S3 bucket
# It requires the following environment variables:
# - ALWAY_PUSH_ALL_FILES: Whether to process all files or only created/modified files (default: false)

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

# Create or clear files_to_modify.txt, files_to_create.txt, and files_to_delete.txt
> /workspace/files_to_modify.txt
> /workspace/files_to_create.txt
> /workspace/files_to_delete.txt

# Cold start scenario: If s3_files.txt doesn't exist, treat all files as created
if [ ! -f /workspace/s3_files.txt ]; then
  echo "Cold start detected (no previous s3_files.txt). Treating all files as created..."
  # Extract filenames from s3_files.txt.new and add to files_to_create.txt
  cat /workspace/s3_files.txt.new | awk '{print $1}' >> /workspace/files_to_create.txt
  echo "Added $(wc -l < /workspace/files_to_create.txt) files to files_to_create.txt"
# Normal case: If s3_files.txt exists, compare and create files_to_modify.txt, files_to_create.txt, and files_to_delete.txt
elif [ -s /workspace/s3_files.txt ]; then
  echo "Checking for modified, created, and deleted files..."
  
  # Find created and modified files
  while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    old_file_exists=$(grep -q "^${file} " /workspace/s3_files.txt && echo "yes" || echo "no")
    
    if [ "$old_file_exists" = "no" ]; then
      # File is newly created
      echo "$file" >> /workspace/files_to_create.txt
    else
      # Check if file was modified
      old_mod_date=$(grep "^${file} " /workspace/s3_files.txt | awk '{print $2}')
      old_mod_time=$(grep "^${file} " /workspace/s3_files.txt | awk '{print $3}')
      new_mod_datetime=$(get_mod_datetime "$file")
      new_mod_date=$(echo "$new_mod_datetime" | awk '{print $1}')
      new_mod_time=$(echo "$new_mod_datetime" | awk '{print $2}')
      
      if [ "$old_mod_date" != "$new_mod_date" ] || [ "$old_mod_time" != "$new_mod_time" ]; then
        echo "$file" >> /workspace/files_to_modify.txt
      fi
    fi
  done < /workspace/s3_files.txt.new
  
  # Find deleted files - read from old file list and check against new file list
  while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    new_file_exists=$(grep -q "^${file} " /workspace/s3_files.txt.new && echo "yes" || echo "no")
    
    if [ "$new_file_exists" = "no" ]; then
      # File was deleted
      echo "$file" >> /workspace/files_to_delete.txt
    fi
  done < /workspace/s3_files.txt
else
  # s3_files.txt exists but is empty
  echo "Previous s3_files.txt is empty. Treating all files as created..."
  # Extract filenames from s3_files.txt.new and add to files_to_create.txt
  cat /workspace/s3_files.txt.new | awk '{print $1}' >> /workspace/files_to_create.txt
  echo "Added $(wc -l < /workspace/files_to_create.txt) files to files_to_create.txt"
fi

# List synced files
echo "Synced files:"
ls -la /workspace/files/

# Print stats
echo "Files to create: $(wc -l < /workspace/files_to_create.txt)"
echo "Files to modify: $(wc -l < /workspace/files_to_modify.txt)"
echo "Files to delete: $(wc -l < /workspace/files_to_delete.txt)"
