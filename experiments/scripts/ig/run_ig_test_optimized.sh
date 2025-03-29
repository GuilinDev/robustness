#!/bin/bash

# Ensure directory structure exists
mkdir -p experiments/results/ig_viz_optimized
mkdir -p experiments/results/figures/ig_optimized

# Install required dependencies
pip install -r experiments/scripts/ig/requirements.txt

# Make sure OpenCV system dependencies are installed
if ! ldconfig -p | grep -q libGL.so.1; then
  echo "Installing system dependencies for OpenCV..."
  apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
fi

# Randomly select 1000 images for testing
echo "Preparing optimized test dataset..."
TEST_DIR="experiments/data/ig_test_subset"
mkdir -p ${TEST_DIR}

# If the subset directory is empty, create a subset with 1000 random images
if [ -z "$(ls -A ${TEST_DIR} 2>/dev/null)" ]; then
  echo "Creating test subset with 1000 random images..."
  # Find all image files
  find experiments/data/tiny-imagenet-200/val -type f -name "*.JPEG" | shuf -n 1000 | while read img; do
    # Create target directory structure identical to the original path
    REL_PATH=$(echo $img | sed "s|experiments/data/tiny-imagenet-200/val/||")
    DIR_NAME=$(dirname ${REL_PATH})
    mkdir -p "${TEST_DIR}/${DIR_NAME}"
    # Copy image to subset directory
    cp "$img" "${TEST_DIR}/${REL_PATH}"
  done
  echo "Test subset created with $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) images"
fi

# Modify a temporary copy of test_ig_robustness.py to reduce n_steps
echo "Creating optimized test script..."
OPTIMIZED_SCRIPT="experiments/scripts/ig/test_ig_robustness_optimized.py"
cp experiments/scripts/ig/test_ig_robustness.py ${OPTIMIZED_SCRIPT}

# Use sed to replace n_steps=50 with n_steps=20
sed -i 's/n_steps=50/n_steps=20/g' ${OPTIMIZED_SCRIPT}
echo "Reduced IG steps from 50 to 20 for better performance"

# Run the optimized IG test
echo "Starting optimized IG test..."
python ${OPTIMIZED_SCRIPT} \
  --image_dir ${TEST_DIR} \
  --output_file experiments/results/ig_robustness_optimized_results.json \
  --temp_file experiments/results/ig_robustness_optimized_results_temp.json \
  --model_type standard \
  --save_viz \
  --viz_dir experiments/results/ig_viz_optimized

# Analyze the results
echo "Analyzing test results..."
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_optimized_results.json \
  --figures_dir experiments/results/figures/ig_optimized \
  --report_path experiments/results/ig_optimized_analysis_report.md \
  --severity_level 3

# Clean up temporary script
rm ${OPTIMIZED_SCRIPT}

echo "Optimized IG robustness test completed!"
echo "Results saved to experiments/results/ig_robustness_optimized_results.json"
echo "Analysis report saved to experiments/results/ig_optimized_analysis_report.md"
echo "Heatmaps saved to experiments/results/figures/ig_optimized/" 