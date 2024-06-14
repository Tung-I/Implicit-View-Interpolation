import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os

im_pair_dir = "data/banj_00_02"
im1_fname = "000_v0_origin.png"
im2_fname = "000_v2_origin.png"
kpts1_fname = "000_v0_origin.dat"
kpts2_fname = "000_v2_origin.dat"
out_dir = "results/triangulation"

def apply_affine_transform(roi, tri_src, tri_dst, dst_shape):
    """
    roi: target region
    tri_src: source triangle vertices
    tri_dst: destination triangle vertices
    dst_shape: shape of the destination region in (H, W)
    """
    mat = cv2.getAffineTransform(tri_src.astype(np.float32), tri_dst.astype(np.float32)) # (2, 3)
    return cv2.warpAffine(roi+1, mat, (dst_shape[1], dst_shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def morph_images(im1, im2, keypoints1, keypoints2, triangles, alpha):
    """Morphs the two images based on the keypoints and triangles.
    """
    morph_image = np.zeros(im1.shape, dtype=im1.dtype)

    for tri_indices in triangles:
        tri1 = keypoints1[tri_indices][..., ::-1]  # Coordinates of the triangle vertices (3, 2)
        tri2 = keypoints2[tri_indices][..., ::-1] 
        tri_morph = (1 - alpha) * tri1 + alpha * tri2
        # # Extract the rectangular RoIs from the images
        r1 = cv2.boundingRect(tri1.astype(np.int32))
        r2 = cv2.boundingRect(tri2.astype(np.int32))
        r_morph = cv2.boundingRect(tri_morph.astype(np.int32))

        roi1 = im1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        roi2 = im2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        # Calculate the offsets between the top left corner of the RoIs and the triangle vertices
        tri1_offset = tri1 - r1[:2]  # (3, 2)
        tri2_offset = tri2 - r2[:2]
        tri_morph_offset = tri_morph - r_morph[:2]
        # print(f"morph area H={r_morph[3]}, morph area W={r_morph[2]}")
        morph_area = morph_image[r_morph[1]:r_morph[1] + r_morph[3], r_morph[0]:r_morph[0] + r_morph[2]]
        warp_roi1 = apply_affine_transform(roi1, tri1_offset, tri_morph_offset, (morph_area.shape[0], morph_area.shape[1]))
        warp_roi2 = apply_affine_transform(roi2, tri2_offset, tri_morph_offset, (morph_area.shape[0], morph_area.shape[1]))
        # print(f"war_roi1 shape: {warp_roi1.shape}, war_roi2 shape: {warp_roi2.shape}")

        # Blending the two warped RoIs
        blended_roi = (1 - alpha) * warp_roi1 + alpha * warp_roi2
        morph_image[r_morph[1]:r_morph[1] + r_morph[3], r_morph[0]:r_morph[0] + r_morph[2]] = blended_roi

    return morph_image

def add_boundary_points(kpts, im_dim):
    h, w = im_dim[:2]
    bound_pts = np.array([[0, 0], [0, w-1], [h-1, 0], [h-1, w-1]])

    edge_x = np.linspace(0, w-1, 10, endpoint=True)
    edge_y = np.linspace(0, h-1, 10, endpoint=True)
    bound_pts = np.vstack([bound_pts,
                        np.stack([np.zeros_like(edge_x), edge_x], axis=-1),
                        np.stack([np.ones_like(edge_x)*(h-1), edge_x], axis=-1),
                        np.stack([edge_y, np.zeros_like(edge_y)], axis=-1),
                        np.stack([edge_y, np.ones_like(edge_y)*(w-1)], axis=-1)])
    return np.vstack([kpts, bound_pts])

def filter_triangles(triangles, kpts, max_area):
    filtered_triangles = []
    for tri_indices in triangles:
        tri = kpts[tri_indices]
        area = 0.5 * abs(np.linalg.det(np.array([
            [tri[0, 0], tri[0, 1], 1],
            [tri[1, 0], tri[1, 1], 1],
            [tri[2, 0], tri[2, 1], 1]   
        ])))
        if max_area is None or area < max_area:
            filtered_triangles.append(tri_indices)
    return np.array(filtered_triangles).astype(np.int32)

def rescale_points(points, image_shape):
    """Rescale keypoints from [-1, 1] to image coordinates.
    """
    height, width = image_shape[:2]
    points[:, 0] = (points[:, 0] + 1) * 0.5 * height
    points[:, 1] = (points[:, 1] + 1) * 0.5 * width
    return points

if __name__ == "__main__":

    im1 = cv2.imread(os.path.join(im_pair_dir, im1_fname))
    im2 = cv2.imread(os.path.join(im_pair_dir, im2_fname))
    kpts1 = np.load(os.path.join(im_pair_dir, kpts1_fname), allow_pickle=True)
    kpts2 = np.load(os.path.join(im_pair_dir, kpts2_fname), allow_pickle=True)
    kpts1 = rescale_points(kpts1, im1.shape)
    kpts2 = rescale_points(kpts2, im2.shape)


# # Save keypoint visualization
# def visualize_keypoints(image, keypoints):
#     for i in range(keypoints.shape[0]):
#         x, y = keypoints[i]
#         cv2.circle(image, (int(y), int(x)), 2, (0, 255, 0), -1)
#     return image

# vis_im1_kpts = visualize_keypoints(im1.copy(), keypoints1)
# vis_im2_kpts = visualize_keypoints(im2.copy(), keypoints2)
# # Save the images
# cv2.imwrite(os.path.join(im_pair_dir, "kps1_vis.png"), vis_im1_kpts)
# cv2.imwrite(os.path.join(im_pair_dir, "kps2_vis.png"), vis_im2_kpts)


    # Triangulation on the first image
    kpts1 = add_boundary_points(kpts1, im1.shape)
    tri = Delaunay(kpts1)
    triangles = tri.simplices  # Indices of the triangle vertices
    print(f"Number of triangles: {triangles.shape[0]}")
    triangles = filter_triangles(triangles, kpts1, max_area=10000)
    print(f"Number of triangles after filtering: {triangles.shape[0]}")

    # Plit triangles on the first image
    plt.figure()
    plt.triplot(kpts1[:, 1], kpts1[:, 0], triangles)
    plt.plot(kpts1[:, 1], kpts1[:, 0], 'o')
    plt.imshow(im1)
    plt.title("Triangulation on the first image")
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "triangles.png"))

    # Morph images for different alpha values
    alphas = [0, 0.25, 0.5, 0.75, 1]
    morph_images_list = [morph_images(im1, im2, kpts1, kpts2, triangles, alpha) for alpha in alphas]

    # Save the results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, alpha in enumerate(alphas):
        cv2.imwrite(os.path.join(out_dir, f"morph_{i}.png"), morph_images_list[i])
