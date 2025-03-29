#!/bin/bash

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$REPO_ROOT" ]; then
    # Fallback if not in a git repository
    REPO_ROOT=$(cd $(dirname $0)/../.. && pwd)
fi

echo "Repository root directory: $REPO_ROOT"

# Create data directory using absolute path
DATA_DIR="$REPO_ROOT/experiments/data"
mkdir -p "$DATA_DIR"
echo "Created data directory: $DATA_DIR"

# Check if dataset already exists
if [ -d "$DATA_DIR/tiny-imagenet-200" ]; then
    echo "Tiny-ImageNet-200 dataset already exists, skipping download and extraction"
else
    echo "Downloading Tiny-ImageNet-200 dataset..."
    wget -P "$DATA_DIR" http://cs231n.stanford.edu/tiny-imagenet-200.zip
    
    if [ $? -ne 0 ]; then
        echo "Download failed, please check your network connection or download manually"
        exit 1
    fi
    
    echo "Extracting Tiny-ImageNet-200 dataset..."
    unzip "$DATA_DIR/tiny-imagenet-200.zip" -d "$DATA_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Extraction failed, please extract the file manually"
        exit 1
    fi
    
    echo "Removing downloaded zip file to save space..."
    rm "$DATA_DIR/tiny-imagenet-200.zip"
    
    # Reorganize training set directory structure
    echo "Reorganizing training set directory structure..."
    
    # Training data
    cd "$DATA_DIR/tiny-imagenet-200/train"
    for DIR in $(ls); do
        cd "$DIR"
        rm -f *.txt
        mv images/* .
        rm -r images
        cd ..
    done
    
    # Validation data
    cd "$DATA_DIR/tiny-imagenet-200/val"
    if [ -d "images" ]; then
        echo "Reorganizing validation set directory structure..."
        annotate_file="val_annotations.txt"
        length=$(cat $annotate_file | wc -l)
        for i in $(seq 1 $length); do
            # Get the i-th line
            line=$(sed -n ${i}p $annotate_file)
            # Get filename and directory name
            file=$(echo $line | cut -f1 -d" " )
            directory=$(echo $line | cut -f2 -d" ")
            mkdir -p $directory
            mv images/$file $directory
        done
        rm -r images
    fi
    
    # Return to repository root
    cd "$REPO_ROOT"
fi

echo "Tiny-ImageNet-200 dataset is ready"
echo "Dataset location: $DATA_DIR/tiny-imagenet-200"

# List data directory contents to confirm setup
echo "Dataset directory contents:"
ls -la "$DATA_DIR/tiny-imagenet-200"

# Count files to verify
echo "Validation set image count:"
find "$DATA_DIR/tiny-imagenet-200/val" -type f -name "*.JPEG" | wc -l

echo "Training set image count:"
find "$DATA_DIR/tiny-imagenet-200/train" -type f -name "*.JPEG" | wc -l

echo "Dataset setup complete!"
