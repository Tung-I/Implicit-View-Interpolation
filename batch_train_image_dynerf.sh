#!/bin/bash

# Define directories and base filenames
CONFIG_DIR="data/dynerf/coffee_martini"
MODEL_DIR="data/dynerf/coffee_martini/image_models"

# Array of image paths
image_paths=("data/dynerf/coffee_martini/cam00.png" "data/dynerf/coffee_martini/cam09.png" "data/dynerf/coffee_martini/cam13.png" "data/dynerf/coffee_martini/cam20.png")

# Generate config files for each image
for image_path in "${image_paths[@]}"; do
    image_basename=$(basename "$image_path" .png)
    config_path="$CONFIG_DIR/$image_basename.yaml"
    echo "Generating config for $image_path"
    python generate_configs_dynerf.py --task image --cfg_dir $CONFIG_DIR --model_dir $MODEL_DIR --im_path $image_path
done

# Run the training script for each config file
for image_path in "${image_paths[@]}"; do
    image_basename=$(basename "$image_path" .png)
    config_path="$CONFIG_DIR/$image_basename.yaml"
    echo "Running training for config $config_path"
    python train_image.py --config $config_path
done
