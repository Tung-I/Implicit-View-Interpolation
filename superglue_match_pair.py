import os 
import cv2
import numpy as np
import argparse
import torch
from models.matching import Matching
from models.utils import frame2tensor
import glob


def visualize_matches(im1, im2, kpts1, kpts2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    matched_image = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    matched_image[:h1, :w1] = im1
    matched_image[:h2, w1:w1+w2] = im2

    for pt1, pt2 in zip(kpts1, kpts2):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + w1), int(pt2[1]))
        cv2.line(matched_image, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(matched_image, pt1, 2, (0, 0, 255), 1)
        cv2.circle(matched_image, pt2, 2, (0, 0, 255), 1)

    return matched_image


if __name__ == '__main__':

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
    im_files = glob.glob(os.path.join(data_dir, "*.png"))
    if len(im_files) != 2:
        raise ValueError("There should be exactly 2 png files in the data directory.")
    im0_fname = os.path.basename(im_files[0])
    im1_fname = os.path.basename(im_files[1])
    n_kpts = args.kpts

    print("Loading images...")
    im0 = cv2.imread(os.path.join(data_dir, im0_fname), cv2.IMREAD_GRAYSCALE)
    im1 = cv2.imread(os.path.join(data_dir, im1_fname), cv2.IMREAD_GRAYSCALE)

    config = {
        # "superpoint": {
        #     "nms_radius": 2,  #smaller value will allow kpts to be closer together {4, 3, 2}
        #     "keypoint_threshold": 0.004,  #smaller value will allow more kpts to be detected {0.005, 0.002, 0.001}
        #     "max_keypoints": n_kpts
        # },
        "superpoint": {
            "nms_radius": 4,  #smaller value will allow kpts to be closer together {4, 3, 2}
            "keypoint_threshold": 0.002,  #smaller value will allow more kpts to be detected {0.005, 0.002, 0.001}
            "max_keypoints": n_kpts
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    print("Loading model and perform matching...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matching = Matching(config).eval().to(device)

    im0_tensor = frame2tensor(im0, device)
    im1_tensor = frame2tensor(im1, device)

    pred = matching({'image0': im0_tensor, 'image1': im1_tensor})

    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    matched_im = visualize_matches(im0, im1, mkpts0, mkpts1)
    save_dir = data_dir
    cv2.imwrite(os.path.join(save_dir, f"{n_kpts}", "matches.png"), matched_im)

    mkpts0 = np.array(mkpts0)
    mkpts1 = np.array(mkpts1)

    # Normalize mkpts in range [-1, 1]
    mkpts0[:, 0] /= im0.shape[1]
    mkpts0[:, 1] /= im0.shape[0]
    mkpts0 = mkpts0 * 2 - 1
    mkpts1[:, 0] /= im1.shape[1]
    mkpts1[:, 1] /= im1.shape[0]
    mkpts1 = mkpts1 * 2 - 1

    # Swap x and y coordinates
    mkpts0 = mkpts0[:, [1, 0]]
    mkpts1 = mkpts1[:, [1, 0]]

    # Save keypoints
    print(f"Save the keypoints to: {os.path.join(save_dir, str(n_kpts))}")
    kpts_fname0 = os.path.join(save_dir, f"{n_kpts}", f"{im0_fname.split('.')[0]}.dat")
    kpts_fname1 = os.path.join(save_dir, f"{n_kpts}", f"{im1_fname.split('.')[0]}.dat")
    mkpts0.dump(kpts_fname0)
    mkpts1.dump(kpts_fname1)

    print("Done.")



