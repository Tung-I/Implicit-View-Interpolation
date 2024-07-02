import cv2
import numpy as np
import os.path as osp
import glob

root_dir = 'data/dynerf/coffee_martini_preprocess'
kpts_dirs = ['cam00_cam09', 'cam00_cam13', 'cam00_cam20', 'cam09_cam13', 'cam09_cam20', 'cam13_cam20']

if __name__ == "__main__":
    for _dir in kpts_dirs:
        kpts_path = osp.join(root_dir, _dir, '4096')
        kpts_files = glob.glob(osp.join(kpts_path, '*.dat'))
        for kpts_file in kpts_files:
            kpts = np.load(kpts_file, allow_pickle=True)
            # Multiply by 2 to get the original image size
            kpts = np.array(kpts) * 2
            # Save the keypoints as a dat file
            cam_id = kpts_file.split('/')[-1].split('.')[0].split('_')[0]
            out_path = osp.join(kpts_path, f'{cam_id}.dat')
            print(f'Saving upsampled keypoints to {out_path}')
            kpts.dump(out_path)

            