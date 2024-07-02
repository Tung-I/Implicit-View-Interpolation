import cv2
import numpy as np
import os.path as osp
import os

data_dir = '/dlbimg/datasets/dynerf/coffee_martini'
cam_ids = [i for i in range(0, 21)]
output_dir = 'data/dynerf/coffee_martini_preprocess'

if __name__ == "__main__":
    for cam_id in cam_ids:
        # for cam_00 to cam_20
        video_path = osp.join(data_dir, f'cam{cam_id:02d}.mp4')
        # Check if video exists
        if not osp.exists(video_path):
            print(f'Video {video_path} does not exist.')
            continue
        # extract the first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        # save the frame
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        output_path = osp.join(output_dir, f'cam{cam_id:02d}.png')
        cv2.imwrite(output_path, frame)
        print(f'Saved frame to {output_path}')

        # Downsample the image
        img = cv2.imread(output_path)
        img = cv2.resize(img, (1352, 1014))
        output_path = osp.join(output_dir, f'cam{cam_id:02d}_downsampled.png')
        cv2.imwrite(output_path, img)
        print(f'Saved downsampled frame to {output_path}')



