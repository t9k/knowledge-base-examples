#!/bin/sh

# This script downloads CAIL2018 files from ModelScope

set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

# Download CAIL2018_ALL_DATA.zip and unzip it
if [ ! -d "law" ]; then
    git clone https://modelsphere.qy.t9kcloud.cn/datasets/t9k-ai/cn-laws.git
    mv cn-laws law
else
    echo "law directory already exists"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
