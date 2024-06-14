# A python scrip that generates a video from a set of images
import os
import numpy as np
import cv2
import argparse
import glob


data_dir = "/dlbimg/datasets/View_transition/content_banjoman_960x540"
tgt_view = "vw_000"
out_dir = "/dlbimg/datasets/View_transition/content_banjoman_960x540/1s_video"
out_fname = "1s.mp4"

if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    view_ids = ['000', '001', '002', '003', '004', '005']

    for view_id in view_ids:
        tgt_view = "vw_" + view_id
        out_fname = f"1s_{tgt_view}.mp4"
        frame_dir = os.path.join(data_dir, tgt_view)
        fnames = glob.glob(os.path.join(frame_dir, "*.png"))
        fnames = sorted(fnames)

        n_frame_for_video = 30

        im_list = []
        for i in range(n_frame_for_video):
            im = cv2.imread(fnames[i])
            im_list.append(im)

        fps = 15

        out_path = os.path.join(out_dir, out_fname)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (im.shape[1], im.shape[0]))
        for i in range(len(im_list)):
            out.write(im_list[i])
        out.release()
        print(f"{tgt_view}; Video generated")


