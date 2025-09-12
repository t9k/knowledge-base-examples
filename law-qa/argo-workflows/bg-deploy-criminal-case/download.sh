#!/bin/sh
set -e

# Log script start with timestamp
echo "$(date +'%Y-%m-%d %H:%M:%S') - Starting download.sh script"

# Clone repository provided as first argument and prepare directory
REPO_URL="${1:-}"

if [ -z "$REPO_URL" ]; then
    echo "Error: repository URL is required as the first argument."
    exit 1
fi

if [ ! -d "criminal-case" ]; then
    git lfs install
    CLONED_DIR=$(basename "$REPO_URL")
    CLONED_DIR=${CLONED_DIR%.git}
    if [ -d "$CLONED_DIR" ]; then
        echo "Found existing directory '$CLONED_DIR', removing it before cloning"
        rm -rf "$CLONED_DIR"
    fi
    echo "Cloning repository: $REPO_URL"
    git clone "$REPO_URL"
    if [ -d "$CLONED_DIR" ]; then
        mv "$CLONED_DIR" criminal-case
    else
        echo "Error: expected cloned directory '$CLONED_DIR' not found."
        exit 1
    fi
else
    echo "criminal-case directory already exists"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') - download.sh completed successfully"
