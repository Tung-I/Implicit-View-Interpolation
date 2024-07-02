#!/bin/bash

# Define directories and base filenames
DATA_DIR="data/dynerf/coffee_martini"
N_KPTS=4096

# Array of image filenames
cams=("00" "09" "13" "20")


# # Generate config files for each pair of images
# for i in {0..3}; do
#     for j in $(seq $((i+1)) 2); do
#         camid1=${cams[i]}
#         camid2=${cams[j]}
#         python generate_configs_dynerf.py --task warping --cfg_dir "$DATA_DIR" --im1 "$camid1" --im2 "$camid2" --kpts $N_KPTS
#     done
# done

# # Run the train_warping.py script for each generated config file
# for i in {0..5}; do
#     for j in $(seq $((i+1)) 5); do
#         im1=${images[i]}
#         im2=${images[j]}
#         config_path="$DATA_DIR/${im1}_${im2}/${N_KPTS}/${im1}_${im2}.yaml"
#         echo "Training warping model: $config_path"
#         python train_warping.py --config "$config_path"
#     done    
# done

# python train_warping.py --config data/dynerf/coffee_martini/cam00_cam09/4096/00_09.yaml
python train_warping.py --config data/dynerf/coffee_martini/cam00_cam13/4096/00_13.yaml
python train_warping.py --config data/dynerf/coffee_martini/cam00_cam20/4096/00_20.yaml
python train_warping.py --config data/dynerf/coffee_martini/cam09_cam13/4096/09_13.yaml
python train_warping.py --config data/dynerf/coffee_martini/cam09_cam20/4096/09_20.yaml
python train_warping.py --config data/dynerf/coffee_martini/cam13_cam20/4096/13_20.yaml