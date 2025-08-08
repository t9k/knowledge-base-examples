#!/bin/sh

# This script downloads cn-judgment-docs files from ModelSphere

set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

rm -rf civil-case
git lfs install
git clone https://modelsphere.qy.t9kcloud.cn/datasets/t9k-ai/cn-judgment-docs-demo.git
mv cn-judgment-docs-demo civil-case/

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
