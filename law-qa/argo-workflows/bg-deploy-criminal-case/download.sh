#!/bin/sh

# This script downloads CAIL2018 files from ModelScope

set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

# Download CAIL2018_ALL_DATA.zip and unzip it
if [ ! -d "criminal-case" ]; then
    mkdir criminal-case && cd criminal-case
    wget https://modelscope.cn/datasets/qazwsxplkj/CAIL2018-demo/resolve/master/exercise_contest_data_valid.json
else
    echo "criminal-case directory already exists"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
