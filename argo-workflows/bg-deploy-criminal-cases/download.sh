#!/bin/sh

# This script downloads CAIL2018 files from Aliyun

set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

# Download CAIL2018_ALL_DATA.zip and unzip it
if [ ! -d "criminal-cases" ]; then
    wget https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip
    unzip CAIL2018_ALL_DATA.zip
    mv final_all_data criminal-cases
    rm CAIL2018_ALL_DATA.zip
else
    echo "criminal-cases directory already exists"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
