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

def initialize_subdiv2d(kpts, image_shape):
    rect = (0, 0, image_shape[1], image_shape[0])
    subdiv = cv2.Subdiv2D(rect)
    
    for p in kpts:
        subdiv.insert((p[1], p[0]))  # Insert using (x, y) format
    
    return subdiv
    
    
def get_triangles(subdiv, kpts):
    triangle_list = subdiv.getTriangleList()
    pt_idx_list = []

    # Map the vertex coordinates back to the original keypoint indices
    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        pt_ids = []
        for x, y in pts:
            # Find the point in keypoints that matches (x, y)
            for i, kp in enumerate(kpts):
                if kp[1] == x and kp[0] == y:
                    pt_ids.append(i)
                    break
        if len(pt_ids) == 3:
            pt_idx_list.append(pt_ids)
    
    return np.array(pt_idx_list)

def rescale_points(points, image_shape):
    """Rescale keypoints from [-1, 1] to image coordinates.
    """
    height, width = image_shape[:2]
    points[:, 0] = (points[:, 0] + 1) * 0.5 * height
    points[:, 1] = (points[:, 1] + 1) * 0.5 * width
    return points

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

if __name__ == "__main__":
    # Load images and keypoints
    im1 = cv2.imread(os.path.join(im_pair_dir, im1_fname))
    im2 = cv2.imread(os.path.join(im_pair_dir, im2_fname))
    kpts1 = np.load(os.path.join(im_pair_dir, kpts1_fname), allow_pickle=True)
    kpts2 = np.load(os.path.join(im_pair_dir, kpts2_fname), allow_pickle=True)
    kpts1 = rescale_points(kpts1, im1.shape)
    kpts2 = rescale_points(kpts2, im2.shape)

    # Add boundary points and initialize Subdiv2D
    subdiv = initialize_subdiv2d(kpts1, im1.shape)

    # Get triangles from Subdiv2D
    triangles = get_triangles(subdiv, kpts1)
    print(f"Number of triangles: {triangles.shape[0]}")

    # Plot triangulation
    plt.figure()
    plt.triplot(kpts1[:, 1], kpts1[:, 0], triangles)
    plt.plot(kpts1[:, 1], kpts1[:, 0], 'o')
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
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
