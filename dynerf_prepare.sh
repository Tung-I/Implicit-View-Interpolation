#!/bin/bash

# Define the directory containing the images
DATA_DIR="/home/tvchen/Implicit-View-Interpolation/data/dynerf/coffee_martini_preprocess"
TGT_DIR="/home/tvchen/Implicit-View-Interpolation/data/dynerf/coffee_martini"
N_KPTS=4096

# Array of camera directories
cams=("cam00" "cam09" "cam13" "cam20")

# Create folders and copy image pairs
for i in {0..3}; do
    for j in $(seq $((i+1)) 3); do
        cam1=${cams[i]}
        cam2=${cams[j]}
        folder_name="${cam1}_${cam2}"
        mkdir -p "$TGT_DIR/$folder_name/4096"
        cp "$DATA_DIR/$folder_name/4096/${cam1}_downsampled.dat" "$TGT_DIR/$folder_name/4096/$cam1.dat"
        cp "$DATA_DIR/$folder_name/4096/${cam2}_downsampled.dat" "$TGT_DIR/$folder_name/4096/$cam2.dat"
        cp "$DATA_DIR/$cam1.png" "$TGT_DIR/$folder_name/$cam1.png"
        cp "$DATA_DIR/$cam2.png" "$TGT_DIR/$folder_name/$cam2.png"
    done
done
