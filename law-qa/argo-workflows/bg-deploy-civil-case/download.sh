#!/bin/sh

# This script downloads CAIL2018 files from ModelScope

set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

# Download CAIL2018_ALL_DATA.zip and unzip it
if [ ! -d "civil-case" ]; then
    mkdir civil-case && cd civil-case
    wget https://modelscope.cn/datasets/qazwsxplkj/cn-judgment-docs-demo/resolve/master/preprocessed_2021_10.csv
else
    echo "civil-case directory already exists"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
