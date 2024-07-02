#!/bin/bash

# Define directories and base filenames
DATA_DIR="data/banjoman"
N_KPTS=2048

# Array of image filenames
images=("000" "001" "002" "003" "004" "005")

# Run the train_warping.py script for each generated config file
for i in {0..5}; do
    for j in $(seq $((i+1)) 5); do
        im1=${images[i]}
        im2=${images[j]}
        config_path="$DATA_DIR/${im1}_${im2}/${N_KPTS}/${im1}_${im2}.yaml"
        echo "Infer warping model: $config_path"
        python infer_warping.py --config "$config_path"
    done    
done
