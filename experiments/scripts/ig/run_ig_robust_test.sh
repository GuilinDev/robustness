#!/bin/bash

# Script configuration - Uncomment and modify these options for optimization
OPTIMIZE=true      # Set to true to enable optimizations
SAMPLE_SIZE=1000   # Number of images to process (set to 0 for all images)
N_STEPS=20         # IG integration steps (original is 50, lower is faster)
RANDOM_SEED=42     # Fixed random seed for reproducibility
# End of configuration

# Ensure directory structure exists
mkdir -p experiments/results/ig_viz_robust
mkdir -p experiments/results/ig_robust_figures
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
  RESULTS_FILE="experiments/results/ig_robustness_robust_results.json"
  TEMP_FILE="experiments/results/ig_robustness_robust_temp.json"
  VIZ_DIR="experiments/results/ig_viz_robust"
  FIGURES_DIR="experiments/results/ig_robust_figures"
  REPORT_PATH="experiments/results/ig_robust_analysis_report.md"
  
  mkdir -p $VIZ_DIR
  mkdir -p $FIGURES_DIR
  
  # Create subset of images if optimizing with sample size
  echo "Preparing optimized dataset with $SAMPLE_SIZE images..."
  TEST_DIR="experiments/data/ig_test_subset"
  mkdir -p ${TEST_DIR}
  
  # Define the sample list file to ensure standard and robust use same samples
  SAMPLE_LIST="experiments/data/samples/ig_sample_list.txt"
  
  # If the sample list doesn't exist, wait for the standard test to create it
  if [ ! -f "$SAMPLE_LIST" ]; then
    echo "Sample list not found. Please run the standard model test script first or manually create the sample list."
    echo "Checking for sample list at: $SAMPLE_LIST"
    exit 1
  fi
  
  # If the subset directory is empty, create it using the same sample list
  if [ -z "$(ls -A ${TEST_DIR} 2>/dev/null)" ]; then
    echo "Creating test subset using the same images as standard model test..."
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
  else
    echo "Using existing test subset with $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) images"
  fi
  
  # Use the subset for testing
  IMAGE_DIR=$TEST_DIR
else
  RESULTS_FILE="experiments/results/ig_robustness_robust_results.json"
  TEMP_FILE="experiments/results/ig_robustness_robust_temp.json"
  VIZ_DIR="experiments/results/ig_viz_robust"
  FIGURES_DIR="experiments/results/ig_robust_figures"
  REPORT_PATH="experiments/results/ig_robust_analysis_report.md"
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

# Run the IG test with robust model
echo "Starting IG robustness test with robust model..."
echo "Image directory: $IMAGE_DIR"
echo "Results will be saved to: $RESULTS_FILE"

python $SCRIPT_PATH \
  --image_dir $IMAGE_DIR \
  --output_file $RESULTS_FILE \
  --temp_file $TEMP_FILE \
  --model_type robust \
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

echo "IG robustness test on robust model completed!"
echo "Results saved to $RESULTS_FILE"
echo "Analysis report saved to $REPORT_PATH"
echo "Heatmaps saved to $FIGURES_DIR" 
