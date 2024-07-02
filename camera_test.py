import os  
import os.path as osp
import numpy as np
import sys
import torch
from scipy.spatial.transform import Slerp, Rotation as R
import cv2
from tqdm import tqdm

root_dir = "/dlbimg/datasets/View_transition/content_banjoman_960x540/colmap"

colmap_scripts_path = "/home/tvchen/colmap/scripts/python"
sys.path.append(colmap_scripts_path)
import read_write_model

def get_camera_pose(image):
    qvec = image.qvec
    tvec = image.tvec
    R = read_write_model.qvec2rotmat(qvec)
    camera_pose = np.hstack((R, tvec.reshape(-1, 1)))
    camera_pose = np.vstack((camera_pose, np.array([0, 0, 0, 1])))
    return camera_pose

def create_intrinsic_matrix(params, model='PINHOLE'):
    if model == 'PINHOLE':
        fx, fy, cx, cy = params
        intrinsic_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
    elif model == 'SIMPLE_PINHOLE':
        f, cx, cy = params
        intrinsic_matrix = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1]
        ])
    else:
        raise NotImplementedError(f"Camera model {model} not implemented.")
    return intrinsic_matrix

def compute_depth_range(image, points3D):
    camera_pose = get_camera_pose(image)
    depths = []
    for point3D_id in image.point3D_ids:
        if point3D_id != -1:
            point3D = points3D[point3D_id]
            point_cam = np.dot(camera_pose[:3, :3], point3D.xyz) + camera_pose[:3, 3]
            depth = point_cam[2]
            depths.append(depth)
    
    return min(depths), max(depths)

def rescale_depth_map(normalized_depth_map, min_depth, max_depth):
    return normalized_depth_map * (max_depth - min_depth) + min_depth

if __name__ == '__main__':
    images = read_write_model.read_images_binary(osp.join(root_dir, 'sparse/0/images.bin'))
    points3d = read_write_model.read_points3D_binary(osp.join(root_dir, 'sparse/0/points3D.bin'))
    cameras = read_write_model.read_cameras_binary(osp.join(root_dir, 'sparse/0/cameras.bin'))

    view1_id = 1
    view2_id = 3
    colmap_im1 = images[view1_id]
    colmap_im2 = images[view2_id]
    colmap_cam1 = cameras[colmap_im1.camera_id]
    colmap_cam2 = cameras[colmap_im2.camera_id]

    im1_K = create_intrinsic_matrix(colmap_cam1.params, colmap_cam1.model)
    im2_K = create_intrinsic_matrix(colmap_cam2.params, colmap_cam2.model)

    im1 = cv2.imread(osp.join(root_dir, 'images', colmap_im1.name)) 
    im2 = cv2.imread(osp.join(root_dir, 'images', colmap_im2.name))
    depth1_path = osp.join("/home/tvchen/Marigold/output/banjo_vw_000/depth_bw/000_pred.png")
    depth2_path = osp.join("/home/tvchen/Marigold/output/banjo_vw_002/depth_bw/000_pred.png")
    depth1 = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)
    depth2 = cv2.imread(depth2_path, cv2.IMREAD_UNCHANGED)

    # Denormalize depth map
    im1_min_depth, im1_max_depth = compute_depth_range(colmap_im1, points3d)
    im2_min_depth, im2_max_depth = compute_depth_range(colmap_im2, points3d)
    denorm_depth1 = rescale_depth_map(depth1 / 255., im1_min_depth, im1_max_depth)
    denorm_depth2 = rescale_depth_map(depth2 / 255., im2_min_depth, im2_max_depth)

    R_t = read_write_model.qvec2rotmat(colmap_im2.qvec)  # (3, 3)
    R_s = read_write_model.qvec2rotmat(colmap_im1.qvec)
    T_t = colmap_im2.tvec  # (3, )
    T_s = colmap_im1.tvec
    R_relative = R_s @ R_t.T
    T_relative = T_s - R_relative @ T_t

    inv_im2_K = np.linalg.inv(im2_K)
    v, u = np.meshgrid(range(im1.shape[0]), range(im1.shape[1]), indexing='ij')
    homogeneous_coords_tgt = np.stack([u, v, np.ones_like(u)], axis=-1)  
    homogeneous_coords_tgt_flat = homogeneous_coords_tgt.reshape(-1, 3).T  # (3, H*W)

    depths = denorm_depth1.flatten()
    valid_depths = depths != 0

    T_relative_outer = np.outer(T_relative, np.array([0, 0, 1]))
    RTz = np.repeat(R_relative[:, :, np.newaxis], depths.size, axis=2)
    print(depths.mean(), depths.std())
    
    RTz -= T_relative_outer[:, :, np.newaxis] / depths  # (3, 3, HW)

    temp_coord = inv_im2_K @ homogeneous_coords_tgt_flat
    temp_coord = np.einsum('ijk, jk -> ik', RTz, temp_coord)
    coords_src = im1_K @ temp_coord
    coords_src /= coords_src[2, :]  # Normalize to make homogeneous


    u_prime, v_prime = coords_src[:2].astype(int)
    valid_indices = (u_prime >= 0) & (u_prime < im1.shape[1]) & (v_prime >= 0) & (v_prime < im1.shape[0]) & valid_depths
    out_im = np.zeros_like(im1)
    out_im[v.flatten()[valid_indices], u.flatten()[valid_indices]] = im1[v_prime[valid_indices], u_prime[valid_indices]]

    out_dir = "output/vis"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(os.path.join(out_dir, "warped.png"), out_im)


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





