import os  
import os.path as osp
import numpy as np
import sys
import torch
from scipy.spatial.transform import Slerp, Rotation as R
import cv2

colmap_scripts_path = "/home/tvchen/colmap/scripts/python"
sys.path.append(colmap_scripts_path)
import read_write_model

def get_camera_pose(image):
    # Extract rotation (quaternion) and translation
    qvec = image.qvec
    tvec = image.tvec
    # Convert quaternion to rotation matrix
    R = read_write_model.qvec2rotmat(qvec)
    # Combine rotation and translation to form the camera pose matrix
    camera_pose = np.hstack((R, tvec.reshape(-1, 1)))
    camera_pose = np.vstack((camera_pose, np.array([0, 0, 0, 1])))
    return camera_pose

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


def interpolate_view(t, img1, img2, cam1, cam2):
    """
    Args:
        t: np.array of shape (N,).
    Returns:
        intp_R: np.array of shape (N, 3, 3).
        intp_T: np.array of shape (N, 3).
        intp_K: np.array of shape
    """
    # Interpolate rotation
    q1 = img1.qvec
    q2 = img2.qvec
    times = np.array([0, 1])
    slerp = Slerp(times, R.from_quat([q1, q2]))
    intp_R = slerp(t) 
    intp_R = intp_R.as_matrix()

    # Interpolate translation
    t1 = img1.tvec
    t2 = img2.tvec
    intp_T = (1 - t) * t1 + t * t2

    # Interpolate intrinsics
    intp_K = (1 - t) * cam1.params + t * cam2.params

    return intp_R, intp_T, intp_K
    
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

def transform_simple_radial_to_pinhole(input_cameras_file, output_cameras_file):
    cameras = read_write_model.read_cameras_binary(input_cameras_file)
    new_cameras = {}

    for cam_id, cam in cameras.items():
        if cam.model == 'SIMPLE_RADIAL':
            f, cx, cy, _ = cam.params  # Ignore the radial distortion parameter k1
            new_params = [f, f, cx, cy]  # Pinhole model parameters: [fx, fy, cx, cy]

            # Create a new Camera instance with the PINHOLE model
            new_cam = read_write_model.Camera(
                id=cam.id,
                model='PINHOLE',
                width=cam.width,
                height=cam.height,
                params=np.array(new_params)
            )
            new_cameras[cam_id] = new_cam
        else:
            # If the camera model is not SIMPLE_RADIAL, keep the original camera
            new_cameras[cam_id] = cam

    # Write the new cameras to the output file
    read_write_model.write_cameras_binary(new_cameras, output_cameras_file)

def warp_image(src, intp_R, intp_T, intp_K, depth):
    """
    Args:
        src: numpy array of shape (H, W, 3) representing the source image.
        intp_R: The interpolated rotation matrix (3, 3).
        intp_T: The interpolated translation vector (3,).
        intp_K: The interpolated intrinsic matrix (3, 3).
        depth: The depth map of the source image (H, W) in the scale output by COLMAP.
    """
    h, w = src.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.flatten(), v.flatten()
    z = depth.flatten()

    # Convert pixel coordinates to normalized device coordinates
    uv1 = np.vstack((u, v, np.ones_like(u)))  # (3, H*W)
    # print(uv1.shape)
    K_inv = np.linalg.inv(intp_K)

    norm_coords = K_inv @ uv1

    # Apply the depth to get the 3D coordinates in the camera frame
    norm_coords *= z

    # Construct the 4x4 transformation matrix
    intp_T = np.reshape(intp_T, (3, 1))
    T_mat = np.hstack((intp_R, intp_T))
    T_mat = np.vstack((T_mat, np.array([0, 0, 0, 1])))

    # Transform the 3D points to the interpolated view
    coords_3D = np.vstack((norm_coords, np.ones_like(u)))
    transformed_coords_3D = T_mat @ coords_3D

    # Project the transformed 3D points back to 2D pixel coordinates
    projected_coords_2D = intp_K @ transformed_coords_3D[:3, :]

    # Normalize by the last coordinate to get the final pixel coordinates
    projected_coords_2D /= projected_coords_2D[2, :]

    # Get the final pixel coordinates
    u_prime, v_prime = projected_coords_2D[:2, :].astype(np.int32)

    # Create the output image
    output = np.zeros_like(src)

    # Map the source image to the output image using the computed coordinates
    for i in range(len(u)):
        if 0 <= u_prime[i] < w and 0 <= v_prime[i] < h:
            output[v_prime[i], u_prime[i]] = src[v[i], u[i]]

    return output


def rescale_depth_map(normalized_depth_map, min_depth, max_depth):
    return normalized_depth_map * (max_depth - min_depth) + min_depth

def normalize_depth_map(depth_map):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    return (depth_map - min_depth) / (max_depth - min_depth)