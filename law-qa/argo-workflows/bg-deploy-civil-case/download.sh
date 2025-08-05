#!/bin/sh

# This script downloads CAIL2018 files from ModelScope

set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

# Download CAIL2018_ALL_DATA.zip and unzip it
if [ ! -d "civil-case" ]; then
    git lfs install
    git clone https://modelsphere.qy.t9kcloud.cn/datasets/xyx/cn-judgment-docs-demo.git
    mv cn-judgment-docs-demo civil-case
else
    echo "civil-case directory already exists"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
