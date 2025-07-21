#!/bin/sh

# This script downloads CAIL2018 files from ModelScope

set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

# Download CAIL2018_ALL_DATA.zip and unzip it
if [ ! -d "law" ]; then
    git clone https://www.modelscope.cn/datasets/qazwsxplkj/cn-laws-demo.git
    mv cn-laws-demo law
else
    echo "law directory already exists"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
