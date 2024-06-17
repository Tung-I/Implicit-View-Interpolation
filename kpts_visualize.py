import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os
import argparse
import glob


def rescale_points(points, image_shape):
    """Rescale keypoints from [-1, 1] to image coordinates.
    """
    height, width = image_shape[:2]
    points[:, 0] = (points[:, 0] + 1) * 0.5 * height
    points[:, 1] = (points[:, 1] + 1) * 0.5 * width
    return points

def visualize_keypoints(image, keypoints):
        for i in range(keypoints.shape[0]):
            x, y = keypoints[i]
            cv2.circle(image, (int(y), int(x)), 2, (0, 255, 0), -1)
        return image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", default=" ", type=str,
        help="Path to the list of image pairs."
    )
    parser.add_argument(
        "--kpts", default=1024, type=int,
        help="Number of keypoints to extract."
    )

    args = parser.parse_args()

    data_dir = args.dir
    n_kpts = args.kpts
    im_files = glob.glob(os.path.join(data_dir, "*.png"))
    if len(im_files) != 2:
        raise ValueError("There should be exactly 2 png files in the data directory.")
    im0_fname = os.path.basename(im_files[0])
    im1_fname = os.path.basename(im_files[1])

    im0 = cv2.imread(os.path.join(data_dir, im0_fname))
    im1 = cv2.imread(os.path.join(data_dir, im1_fname))
    kpts_dir = os.path.join(data_dir, f"{n_kpts}")

    kpts0 = np.load(os.path.join(kpts_dir, f"{im0_fname.split('.')[0]}.dat"), allow_pickle=True)
    kpts1 = np.load(os.path.join(kpts_dir, f"{im1_fname.split('.')[0]}.dat"), allow_pickle=True)
    # print(kpts0[:5])

    kpts0 = rescale_points(kpts0, im0.shape)
    kpts1 = rescale_points(kpts1, im1.shape)

    # print(kpts0[:5])

    # Save keypoint visualization
    vis_im0_kpts = visualize_keypoints(im0.copy(), kpts0)
    vis_im1_kpts = visualize_keypoints(im1.copy(), kpts1)
    # Save the images
    cv2.imwrite(os.path.join(kpts_dir, "kps0_vis.png"), vis_im0_kpts)
    cv2.imwrite(os.path.join(kpts_dir, "kps1_vis.png"), vis_im1_kpts)