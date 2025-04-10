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

# Create or clear modified_files.txt, created_files.txt, and deleted_files.txt
> /workspace/modified_files.txt
> /workspace/created_files.txt
> /workspace/deleted_files.txt

# Cold start scenario: If s3_files.txt doesn't exist, treat all files as created
if [ ! -f /workspace/s3_files.txt ]; then
  echo "Cold start detected (no previous s3_files.txt). Treating all files as created..."
  # Extract filenames from s3_files.txt.new and add to created_files.txt
  cat /workspace/s3_files.txt.new | awk '{print $1}' >> /workspace/created_files.txt
  echo "Added $(wc -l < /workspace/created_files.txt) files to created_files.txt"
# Normal case: If s3_files.txt exists, compare and create modified_files.txt, created_files.txt, and deleted_files.txt
elif [ -s /workspace/s3_files.txt ]; then
  echo "Checking for modified, created, and deleted files..."
  
  # Find created and modified files
  while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    old_file_exists=$(grep -q "^${file} " /workspace/s3_files.txt && echo "yes" || echo "no")
    
    if [ "$old_file_exists" = "no" ]; then
      # File is newly created
      echo "$file" >> /workspace/created_files.txt
    else
      # Check if file was modified
      old_mod_date=$(grep "^${file} " /workspace/s3_files.txt | awk '{print $2}')
      old_mod_time=$(grep "^${file} " /workspace/s3_files.txt | awk '{print $3}')
      new_mod_datetime=$(get_mod_datetime "$file")
      new_mod_date=$(echo "$new_mod_datetime" | awk '{print $1}')
      new_mod_time=$(echo "$new_mod_datetime" | awk '{print $2}')
      
      if [ "$old_mod_date" != "$new_mod_date" ] || [ "$old_mod_time" != "$new_mod_time" ]; then
        echo "$file" >> /workspace/modified_files.txt
      fi
    fi
  done < /workspace/s3_files.txt.new
  
  # Find deleted files - read from old file list and check against new file list
  while read -r line; do
    file=$(echo "$line" | awk '{print $1}')
    new_file_exists=$(grep -q "^${file} " /workspace/s3_files.txt.new && echo "yes" || echo "no")
    
    if [ "$new_file_exists" = "no" ]; then
      # File was deleted
      echo "$file" >> /workspace/deleted_files.txt
    fi
  done < /workspace/s3_files.txt
else
  # s3_files.txt exists but is empty
  echo "Previous s3_files.txt is empty. Treating all files as created..."
  # Extract filenames from s3_files.txt.new and add to created_files.txt
  cat /workspace/s3_files.txt.new | awk '{print $1}' >> /workspace/created_files.txt
  echo "Added $(wc -l < /workspace/created_files.txt) files to created_files.txt"
fi

# Compare old s3_files.txt with dify_docs.json file list
if [ -f /workspace/s3_files.txt ] && [ -f /workspace/dify_docs.json ]; then
  echo "Comparing s3_files.txt with dify_docs.json..."
  
  # Create temporary files with just the filenames for comparison
  grep -v "^$" /workspace/s3_files.txt | awk '{print $1}' | sort > /workspace/s3_files_names.tmp
  cat /workspace/dify_docs.json | jq -r '.data[].name' | sort > /workspace/dify_docs_names.tmp
  
  # Check if files are missing in either list
  if ! diff -q /workspace/s3_files_names.tmp /workspace/dify_docs_names.tmp > /dev/null; then
    echo "WARNING: Inconsistency found between s3_files.txt and dify_docs.json"
    echo "Files in s3 but not in Dify:"
    comm -23 /workspace/s3_files_names.tmp /workspace/dify_docs_names.tmp
    echo "Files in Dify but not in s3:"
    comm -13 /workspace/s3_files_names.tmp /workspace/dify_docs_names.tmp
  else
    echo "s3_files.txt and dify_docs.json are consistent"
  fi
  
  # Clean up temporary files
  rm -f /workspace/s3_files_names.tmp /workspace/dify_docs_names.tmp
fi

# Move new s3_files.txt into place
mv /workspace/s3_files.txt.new /workspace/s3_files.txt

# List synced files
echo "Synced files:"
ls -la /workspace/files/

# Print stats
echo "Created files: $(wc -l < /workspace/created_files.txt)"
echo "Modified files: $(wc -l < /workspace/modified_files.txt)"
echo "Deleted files: $(wc -l < /workspace/deleted_files.txt)"
