#!/bin/bash

# Script configuration - Uncomment and modify these options for optimization
OPTIMIZE=false      # Set to true to enable optimizations
SAMPLE_SIZE=0   # Number of images to process (set to 0 for all images)
N_STEPS=20         # IG integration steps (original is 50, lower is faster)
RANDOM_SEED=42     # Fixed random seed for reproducibility
# End of configuration

# Ensure directory structure exists
mkdir -p experiments/results/ig_viz
mkdir -p experiments/results/ig_standard_figures
mkdir -p experiments/data/samples

# Install required dependencies
pip install -r experiments/scripts/ig/requirements.txt

# Ensure OpenCV system dependencies are installed
if ! ldconfig -p | grep -q libGL.so.1 2>/dev/null; then
  echo "Installing system dependencies for OpenCV..."
  apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
fi

# Create output filenames based on optimization settings
if [ "$OPTIMIZE" = true ] && [ $SAMPLE_SIZE -gt 0 ]; then
  RESULTS_FILE="experiments/scripts/ig/ig_robustness_standard_results.json"
  TEMP_FILE="experiments/scripts/ig/ig_robustness_standard_temp.json"
  VIZ_DIR="experiments/scripts/ig/ig_viz_standard"
  FIGURES_DIR="experiments/scripts/ig/results/ig_standard_figures"
  REPORT_PATH="experiments/scripts/ig/results/ig_standard_analysis_report.md"
  
  mkdir -p $VIZ_DIR
  mkdir -p $FIGURES_DIR
  
  # Create subset of images if optimizing with sample size
  echo "Preparing optimized dataset with $SAMPLE_SIZE images..."
  TEST_DIR="experiments/data/ig_test_subset"
  mkdir -p ${TEST_DIR}
  
  # Define the sample list file to ensure standard and robust use same samples
  SAMPLE_LIST="experiments/data/samples/ig_sample_list.txt"
  
  # If the subset directory is empty or sample list doesn't exist, create a subset with random images
  if [ -z "$(ls -A ${TEST_DIR} 2>/dev/null)" ] || [ ! -f "$SAMPLE_LIST" ]; then
    echo "Creating test subset with $SAMPLE_SIZE random images using seed $RANDOM_SEED..."
    # Set random seed for reproducibility
    export RANDOM=$RANDOM_SEED
    
    # Find all image files
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
    echo "Test subset created with $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) images"
    echo "Sample list saved to $SAMPLE_LIST for reproducibility"
  else
    echo "Using existing test subset with $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) images"
    echo "Sample list already exists at $SAMPLE_LIST"
  fi
  
  # Use the subset for testing
  IMAGE_DIR=$TEST_DIR
else
  RESULTS_FILE="experiments/scripts/ig/ig_robustness_standard_results.json"
  TEMP_FILE="experiments/scripts/ig/ig_robustness_standard_temp.json"
  VIZ_DIR="experiments/scripts/ig/ig_viz_standard"
  FIGURES_DIR="experiments/scripts/ig/results/ig_standard_figures"
  REPORT_PATH="experiments/scripts/ig/results/ig_standard_analysis_report.md"
  IMAGE_DIR="experiments/data/tiny-imagenet-200/val"
  
  mkdir -p $VIZ_DIR
  mkdir -p $FIGURES_DIR
fi

# If optimizing, create temporary script with modified n_steps
if [ "$OPTIMIZE" = true ] && [ $N_STEPS -gt 0 ]; then
  echo "Creating optimized script with $N_STEPS integration steps..."
  SCRIPT_PATH="experiments/scripts/ig/test_ig_robustness_temp.py"
  cp experiments/scripts/ig/test_ig_robustness.py $SCRIPT_PATH
  
  # Use sed to replace n_steps value
  sed -i "s/n_steps=50/n_steps=$N_STEPS/g" $SCRIPT_PATH
  echo "Reduced IG steps from 50 to $N_STEPS for better performance"
else
  SCRIPT_PATH="experiments/scripts/ig/test_ig_robustness.py"
fi

# Run the IG test
echo "Starting IG robustness test with standard model..."
echo "Image directory: $IMAGE_DIR"
echo "Results will be saved to: $RESULTS_FILE"

python $SCRIPT_PATH \
  --image_dir $IMAGE_DIR \
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

# Clean up temporary script if created
if [ "$OPTIMIZE" = true ] && [ $N_STEPS -gt 0 ]; then
  rm $SCRIPT_PATH
fi

echo "IG robustness test on standard model completed!"
echo "Results saved to $RESULTS_FILE"
echo "Analysis report saved to $REPORT_PATH"
echo "Heatmaps saved to $FIGURES_DIR" 