#!/bin/bash

# Create data directory
mkdir -p experiments/data

# Check if dataset already exists
if [ -d "experiments/data/tiny-imagenet-200" ]; then
    echo "Tiny-ImageNet-200 dataset already exists, skipping download and extraction"
else
    echo "Downloading Tiny-ImageNet-200 dataset..."
    wget -P experiments/data http://cs231n.stanford.edu/tiny-imagenet-200.zip
    
    if [ $? -ne 0 ]; then
        echo "Download failed, please check your network connection or download manually"
        exit 1
    fi
    
    echo "Extracting Tiny-ImageNet-200 dataset..."
    unzip experiments/data/tiny-imagenet-200.zip -d experiments/data
    
    if [ $? -ne 0 ]; then
        echo "Extraction failed, please extract the file manually"
        exit 1
    fi
    
    echo "Removing downloaded zip file to save space..."
    rm experiments/data/tiny-imagenet-200.zip
    
    # Reorganize training set directory structure
    echo "Reorganizing training set directory structure..."
    current="$(pwd)/experiments/data/tiny-imagenet-200"
    
    # Training data
    cd $current/train
    for DIR in $(ls); do
        cd $DIR
        rm -f *.txt
        mv images/* .
        rm -r images
        cd ..
    done
    cd $current
    
    # Validation data
    cd $current/val
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
    
    # Return to original directory
    cd $(pwd)
fi

# Check if validation set is correctly structured
if [ ! -d "experiments/data/tiny-imagenet-200/val/images" ]; then
    echo "Tiny-ImageNet-200 validation set directory structure is not in original format, skipping processing"
else
    echo "Validation set directory structure is in original format"
fi

echo "Tiny-ImageNet-200 dataset is ready"
echo "Dataset location: $(pwd)/experiments/data/tiny-imagenet-200"

# List data directory contents to confirm setup
echo "Dataset directory contents:"
ls -la experiments/data/tiny-imagenet-200

# Count files to verify
echo "Validation set image count:"
find experiments/data/tiny-imagenet-200/val -type f -name "*.JPEG" | wc -l

echo "Training set image count:"
find experiments/data/tiny-imagenet-200/train -type f -name "*.JPEG" | wc -l

echo "Dataset setup complete!"
