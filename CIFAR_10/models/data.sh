#!/bin/bash

# Set variables
CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DOWNLOAD_DIR="./cifar-10-batches-py"
TAR_FILE="cifar-10-python.tar.gz"

# Create the directory if it doesn't exist
if [ ! -d "$DOWNLOAD_DIR" ]; then
  mkdir -p "$DOWNLOAD_DIR"
fi

# Download the CIFAR-10 dataset using curl
echo "Downloading CIFAR-10 dataset..."
curl -o $TAR_FILE $CIFAR10_URL

# Check if the download was successful
if [ $? -ne 0 ]; then
  echo "Download failed!"
  exit 1
fi

# Extract the dataset
echo "Extracting CIFAR-10 dataset..."
tar -xzf $TAR_FILE -C $DOWNLOAD_DIR

# Clean up
echo "Cleaning up..."
rm $TAR_FILE

echo "CIFAR-10 dataset downloaded and extracted successfully."
