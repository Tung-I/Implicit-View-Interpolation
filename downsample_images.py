import cv2
import os
import numpy as np
from tqdm import tqdm


def downsample_images(input_dir, output_dir, scale_factor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Downsampling images from {input_dir} to {output_dir} with scale factor {scale_factor}.")
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(input_dir, file))
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.imwrite(os.path.join(output_dir, file), img)

def downsample_banjo(data_dir, output_dir, scale_factor):
    # the images are within data_dir/vw_001, data_dir/vw_002, etc.
    for subdir in os.listdir(data_dir):
        if not 'vw_00' in subdir:
            continue
        downsample_images(os.path.join(data_dir, subdir), os.path.join(output_dir, subdir), scale_factor)

if __name__ == "__main__":
    input_dir = "/dlbimg/datasets/View_transition/content_banjoman_960x540"
    output_dir = "/dlbimg/datasets/View_transition/content_banjoman_480x270"
    scale_factor = 0.5
    downsample_banjo(input_dir, output_dir, scale_factor)

    print("Images downsampled successfully.")