import os  
import os.path as osp
import numpy as np
import sys
import torch
from scipy.spatial.transform import Slerp, Rotation as R
import cv2

from colmap_camera_utils import * 

root_dir = "/dlbimg/datasets/View_transition/content_banjoman_960x540/colmap"

if __name__ == '__main__':
    # cameras = read_write_model.read_cameras_binary(osp.join(root_dir, 'sparse/0/cameras.bin'))
    images = read_write_model.read_images_binary(osp.join(root_dir, 'sparse/0/images.bin'))
    points3d = read_write_model.read_points3D_binary(osp.join(root_dir, 'sparse/0/points3D.bin'))

    transform_simple_radial_to_pinhole(osp.join(root_dir, 'sparse/0/cameras.bin'), osp.join(root_dir, 'sparse/0/cameras_pinhole.bin'))
    cameras = read_write_model.read_cameras_binary(osp.join(root_dir, 'sparse/0/cameras_pinhole.bin'))
    
    t = 0.1
    view1_id = 1
    view2_id = 3
    im1 = images[view1_id]
    im2 = images[view2_id]
    cam1 = cameras[im1.camera_id]
    cam2 = cameras[im2.camera_id]

    intp_R, intp_T, intp_K = interpolate_view(t, im1, im2, cam1, cam2)
    cam_mod = cam1.model
    recon_intp_K = create_intrinsic_matrix(intp_K, cam_mod)

    src1 = cv2.imread(osp.join(root_dir, 'images', im1.name)) 
    src2 = cv2.imread(osp.join(root_dir, 'images', im2.name))

    depth1_path = osp.join("/home/tvchen/Marigold/output/banjo_vw_000/depth_bw/000_pred.png") # (H, W)
    depth2_path = osp.join("/home/tvchen/Marigold/output/banjo_vw_002/depth_bw/000_pred.png")
    depth1 = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)
    depth2 = cv2.imread(depth2_path, cv2.IMREAD_UNCHANGED)

    im1_min_depth, im1_max_depth = compute_depth_range(im1, points3d)
    im2_min_depth, im2_max_depth = compute_depth_range(im2, points3d)

    rescale_depth1 = rescale_depth_map(depth1 / 255., im1_min_depth, im1_max_depth)
    rescale_depth2 = rescale_depth_map(depth2 / 255., im2_min_depth, im2_max_depth)

    warped_im1 = warp_image(src1, intp_R, intp_T, recon_intp_K, rescale_depth1)
    warped_im2 = warp_image(src2, intp_R, intp_T, recon_intp_K, rescale_depth2)

    out_dir = "output/vis"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(os.path.join(out_dir, "warped_im1.png"), warped_im1)
    cv2.imwrite(os.path.join(out_dir, "warped_im2.png"), warped_im2)


    # # Sanity check
    # out_dir = "output/vis"
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # cv2.imwrite(os.path.join(out_dir, "rescale_depth1.png"), (normalize_depth_map(rescale_depth1) * 255).astype(np.uint8))
    # cv2.imwrite(os.path.join(out_dir, "rescale_depth2.png"), (normalize_depth_map(rescale_depth2) * 255).astype(np.uint8))





# class ViewSynthesisLoss(nn.Module):
#     def __init__(self, K1, K2, R, t, z1):
#         super(ViewSynthesisLoss, self).__init__()
#         self.K1 = torch.tensor(K1, dtype=torch.float32)
#         self.K2 = torch.tensor(K2, dtype=torch.float32)
#         self.R = torch.tensor(R, dtype=torch.float32)
#         self.t = torch.tensor(t, dtype=torch.float32)
#         self.z1 = torch.tensor(z1, dtype=torch.float32)
#         self.n = torch.tensor([0, 0, 1], dtype=torch.float32).view(3, 1)
    
#     def forward(self, u, v, t, u_prime, v_prime):
#         # Convert (u, v) to homogeneous coordinates
#         uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).view(-1, 3, 1)
#         uv2 = torch.stack([u_prime, v_prime, torch.ones_like(u_prime)], dim=-1).view(-1, 3, 1)
        
#         # Compute the transformation matrix
#         K1_inv = torch.inverse(self.K1)
#         transformation_matrix = self.K1 @ (self.R - self.t @ self.n.T / self.z1) @ K1_inv
        
#         # Apply the transformation
#         uv1_transformed = transformation_matrix @ uv1
#         uv1_transformed /= uv1_transformed[:, -1, :]  # Convert back to non-homogeneous coordinates
        
#         # Compute the loss
#         loss = nn.MSELoss()(uv1_transformed[:, :2, 0], uv2[:, :2, 0])
#         return loss





