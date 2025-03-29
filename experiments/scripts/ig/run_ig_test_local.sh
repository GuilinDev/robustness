#!/bin/bash

# Local test configuration 
SAMPLE_SIZE=10     # Small number of images for quick local testing
N_STEPS=10         # Reduced integration steps for faster results
RANDOM_SEED=42     # Fixed random seed for reproducibility

# Ensure directory structure exists
mkdir -p experiments/results/ig_viz_local
mkdir -p experiments/results/figures/ig_local
mkdir -p experiments/data/samples

# Install required dependencies
pip install -r experiments/scripts/ig/requirements.txt

# Create test subset for local testing
echo "Preparing local test dataset with $SAMPLE_SIZE images..."
TEST_DIR="experiments/data/ig_test_local"
mkdir -p ${TEST_DIR}

# Define the sample list file 
SAMPLE_LIST="experiments/data/samples/ig_local_sample_list.txt"

# If the subset directory is empty or sample list doesn't exist, create a subset with random images
if [ -z "$(ls -A ${TEST_DIR} 2>/dev/null)" ] || [ ! -f "$SAMPLE_LIST" ]; then
  echo "Creating local test subset with $SAMPLE_SIZE random images using seed $RANDOM_SEED..."
  # Set random seed for reproducibility
  export RANDOM=$RANDOM_SEED
  
  # Find all image files and create sample list
  find experiments/data/tiny-imagenet-200/val -type f -name "*.JPEG" | sort | shuf -n $SAMPLE_SIZE > "$SAMPLE_LIST"
  
  # Create the subset based on the sample list
  cat "$SAMPLE_LIST" | while read img; do
    # Create target directory structure identical to the original path
    REL_PATH=$(echo $img | sed "s|experiments/data/tiny-imagenet-200/val/||")
    DIR_NAME=$(dirname ${REL_PATH})
    mkdir -p "${TEST_DIR}/${DIR_NAME}"
    # Copy image to subset directory
    cp "$img" "${TEST_DIR}/${REL_PATH}"
  done
  echo "Local test subset created with $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) images"
  echo "Sample list saved to $SAMPLE_LIST for reproducibility"
else
  echo "Using existing local test subset with $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) images"
  echo "Sample list already exists at $SAMPLE_LIST"
fi

# Define output filenames
RESULTS_FILE="experiments/results/ig_robustness_local_results.json"
TEMP_FILE="experiments/results/ig_robustness_local_temp.json"
VIZ_DIR="experiments/results/ig_viz_local"
FIGURES_DIR="experiments/results/figures/ig_local"
REPORT_PATH="experiments/results/ig_local_analysis_report.md"

# Create temporary script with modified n_steps
echo "Creating local test script with $N_STEPS integration steps..."
SCRIPT_PATH="experiments/scripts/ig/test_ig_robustness_local.py"
cp experiments/scripts/ig/test_ig_robustness.py $SCRIPT_PATH

# Use sed to replace n_steps value
sed -i "s/n_steps=50/n_steps=$N_STEPS/g" $SCRIPT_PATH
echo "Reduced IG steps from 50 to $N_STEPS for better performance"

# Run the IG local test
echo "Starting IG local test..."
echo "Image directory: $TEST_DIR"
echo "Results will be saved to: $RESULTS_FILE"

python $SCRIPT_PATH \
  --image_dir $TEST_DIR \
  --output_file $RESULTS_FILE \
  --temp_file $TEMP_FILE \
  --model_type standard \
  --save_viz \
  --viz_dir $VIZ_DIR

# Analyze results
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path $RESULTS_FILE \
  --figures_dir $FIGURES_DIR \
  --report_path $REPORT_PATH \
  --severity_level 3

# Clean up temporary script
rm $SCRIPT_PATH

echo "IG local test completed!"
echo "Results saved to $RESULTS_FILE"
echo "Analysis report saved to $REPORT_PATH"
echo "Heatmaps saved to $FIGURES_DIR" 