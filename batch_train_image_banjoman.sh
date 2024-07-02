#!/bin/bash

# Define directories and base filenames
CONFIG_DIR="data/banjoman"
MODEL_DIR="data/banjoman/image_models"
IMAGE_PREFIX="data/banjoman/vw_"
IMAGE_SUFFIX="_000.png"
CONFIG_SUFFIX="_000.yaml"

# Loop over the range of image indices
for i in {0..5}; do
    # Format the image path
    IMAGE_PATH="${IMAGE_PREFIX}$(printf "%03d" $i)${IMAGE_SUFFIX}"
    
    # Generate config file
    python generate_configs.py --task image --cfg_dir $CONFIG_DIR --model_dir $MODEL_DIR --im_path $IMAGE_PATH
done

# Loop over the range of config files
for i in {0..5}; do
    echo "Training image model: ${CONFIG_DIR}/vw_$(printf "%03d" $i)${CONFIG_SUFFIX}"
    # Format the config file path
    CONFIG_PATH="${CONFIG_DIR}/vw_$(printf "%03d" $i)${CONFIG_SUFFIX}"
    
    # Run the training script with the generated config file
    python train_image.py --config $CONFIG_PATH
done