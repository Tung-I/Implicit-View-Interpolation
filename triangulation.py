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

# Load the images
im1 = cv2.imread(os.path.join(im_pair_dir, im1_fname))
im2 = cv2.imread(os.path.join(im_pair_dir, im2_fname))

# Load the matching keypoints from .dat file
kpts1 = np.load(os.path.join(im_pair_dir, kpts1_fname), allow_pickle=True)
kpts2 = np.load(os.path.join(im_pair_dir, kpts2_fname), allow_pickle=True)

# Rescale keypoints from [-1, 1] to image coordinates
def rescale_keypoints(keypoints, image_shape):
    height, width = image_shape[:2]
    keypoints[:, 0] = (keypoints[:, 0] + 1) * 0.5 * height
    keypoints[:, 1] = (keypoints[:, 1] + 1) * 0.5 * width
    return keypoints

keypoints1 = rescale_keypoints(kpts1, im1.shape)
keypoints2 = rescale_keypoints(kpts2, im2.shape)


# # Visualize keypoints on the images
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


# Perform Delaunay triangulation on the keypoints of the first image
tri = Delaunay(keypoints1)

# Get the triangles and their vertex indices
triangles = tri.simplices

# Function to apply affine transformation to each triangle
def apply_affine_transform(src, dst, tri_src, tri_dst, size):
    # Compute the affine transformation matrix
    mat = cv2.getAffineTransform(tri_src.astype(np.float32), tri_dst.astype(np.float32))
    # Apply the affine transformation
    return cv2.warpAffine(src, mat, (size[1], size[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# Perform the interpolation for morphing
def morph_images(im1, im2, keypoints1, keypoints2, triangles, alpha):
    morph_image = np.zeros(im1.shape, dtype=im1.dtype)

    for tri_indices in triangles:
        # Extract vertex coordinates for the current triangle in both images
        tri1 = keypoints1[tri_indices]
        tri2 = keypoints2[tri_indices]
        tri_morph = (1 - alpha) * tri1 + alpha * tri2

        # Define bounding rectangles for the current triangle
        r1 = cv2.boundingRect(tri1)
        r2 = cv2.boundingRect(tri2)
        r_morph = cv2.boundingRect(tri_morph)

        # Offset the triangle vertices by the bounding rectangle top-left corner
        tri1_offset = tri1 - r1[:2]
        tri2_offset = tri2 - r2[:2]
        tri_morph_offset = tri_morph - r_morph[:2]

        # Extract the regions of interest (ROIs) from the images
        roi1 = im1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        roi2 = im2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        # Apply affine transformation to the ROIs
        warp_roi1 = apply_affine_transform(roi1, roi1, tri1_offset, tri_morph_offset, (r_morph[3], r_morph[2]))
        warp_roi2 = apply_affine_transform(roi2, roi2, tri2_offset, tri_morph_offset, (r_morph[3], r_morph[2]))

        # Blend the transformed ROIs
        blended_roi = (1 - alpha) * warp_roi1 + alpha * warp_roi2

        # Place the blended ROI into the morph image
        morph_image[r_morph[1]:r_morph[1] + r_morph[3], r_morph[0]:r_morph[0] + r_morph[2]] = blended_roi

    return morph_image

# Morph images for different alpha values
alphas = [0, 0.25, 0.5, 0.75, 1]
morph_images_list = [morph_images(im1, im2, keypoints1, keypoints2, triangles, alpha) for alpha in alphas]

# Display the results
for i, alpha in enumerate(alphas):
    plt.subplot(1, len(alphas), i+1)
    plt.imshow(cv2.cvtColor(morph_images_list[i], cv2.COLOR_BGR2RGB))
    plt.title(f'Alpha={alpha}')
    plt.axis('off')
plt.show()
