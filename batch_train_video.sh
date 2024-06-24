#!/bin/bash

CFG_OUTPUT_DIR=configs/train/video
FRAME_DATA_DIR=/dlbimg/datasets/View_transition/content_banjoman_480x270

# for v in {0..5}; do
for v in {0..0}; do
    # Make a directory for each video if the directory does not exist
    mkdir -p $CFG_OUTPUT_DIR
    for s in {0..19}; do
        echo "Generating config for video $v, seg_idx $s"
        python generate_configs.py --task neural_video --cfg_out_dir $CFG_OUTPUT_DIR --model_saved_dir output/NIM/video_v${v}_s${s} --data_dir ${FRAME_DATA_DIR}/vw_00${v}  --seg_idx $s --output_file video_v${v}_s${s}.yaml --ckpt_path output/NIM/image_v${v}_s${s}/checkpoints/model_best.pth
        echo "Training temporal warping for video $v, seg_idx $s"
        python train_video.py --config $CFG_OUTPUT_DIR/video_v${v}_s${s}.yaml
    done
done
