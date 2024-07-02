#!/bin/bash

# Define directories and base filenames
DATA_DIR="data/banjoman"
N_KPTS=2048

# Array of image filenames
images=("000" "001" "002" "003" "004" "005")


# Generate config files for each pair of images
for i in {0..5}; do
    for j in $(seq $((i+1)) 5); do
        im1=${images[i]}
        im2=${images[j]}
        python generate_configs.py --task warping --cfg_dir "$DATA_DIR" --im1 "$im1" --im2 "$im2" --kpts $N_KPTS
    done
done

# Run the train_warping.py script for each generated config file
for i in {0..5}; do
    for j in $(seq $((i+1)) 5); do
        im1=${images[i]}
        im2=${images[j]}
        config_path="$DATA_DIR/${im1}_${im2}/${N_KPTS}/${im1}_${im2}.yaml"
        echo "Training warping model: $config_path"
        python train_warping.py --config "$config_path"
    done    
done
