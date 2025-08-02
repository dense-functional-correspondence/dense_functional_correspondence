import trimesh
import numpy as np
import os
from scipy.spatial import KDTree, cKDTree
from PIL import Image, ImageOps
import random
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import time
import json
from concurrent.futures import ProcessPoolExecutor
import cv2

mode = "seen"
folder = "test_annotations"
N_TRIALS = 6
base_dir = "<#TODO: path to the base directory for data>"
MESH_PATH = f"{base_dir}/objaverse_test/{mode}" # this is supposed to contain the extracted pointclouds.
MESH_FULL_PATH = f"{base_dir}/<#TODO: folder to rendered images, which contains the visible_pt_dict and K, RT for cameras>"
RESIZED_IMAGE_PATH = f"{base_dir}/<#TODO: folder to resized and processed images>"
ANNO_PATH = f"{base_dir}/annotations/{folder}/labeled_transforms/{mode}"
OUT_DIR = f"{base_dir}/annotations/{folder}/2d_annotations/{mode}"
VIZ_OUT_DIR = f"{base_dir}/annotations/{folder}/2d_annotations_viz/{mode}"
MASK_OUT_DIR = f"{base_dir}/annotations/{folder}/2d_annotations_masks/{mode}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)
colormap = Image.open("./ziegler_colormap.png").convert("RGB")


def points_in_bounding_box(point_cloud, bbox_vertices):
    # Create a convex hull for the bounding box
    bbox_hull = trimesh.Trimesh(vertices=bbox_vertices).convex_hull
    # Use `contains` method to check if points are inside the bounding box
    contained = bbox_hull.contains(point_cloud)
    # Get the indices of points that are inside the bounding box
    indices = np.where(contained)[0]
    return indices

def load_meshes_obj(action, obj_pair, label_idx, debug = False):
    obj1_name, obj2_name = obj_pair.split("|||")

    obj1_pose = f"{ANNO_PATH}/{action}/{obj_pair}/{label_idx:04d}/{obj1_name}.npy"
    obj2_pose = f"{ANNO_PATH}/{action}/{obj_pair}/{label_idx:04d}/{obj2_name}.npy"
    
    # loading annotated poses
    obj1_pose = np.load(obj1_pose)
    obj2_pose = np.load(obj2_pose)

    # loading pointclouds
    obj1_uid = obj1_name.split("---")[0]
    obj1_pc = np.load(f"{MESH_PATH}/{obj1_name}/{obj1_uid}.npy")[:,:3]
    obj2_uid = obj2_name.split("---")[0]
    obj2_pc = np.load(f"{MESH_PATH}/{obj2_name}/{obj2_uid}.npy")[:,:3]

    obj1_pc = trimesh.PointCloud(obj1_pc)
    obj2_pc = trimesh.PointCloud(obj2_pc)
    
    # loading boxes
    bbox1_p = f"{ANNO_PATH}/{action}/{obj_pair}/{label_idx:04d}/{obj1_name}_bbox.obj"
    bbox2_p = f"{ANNO_PATH}/{action}/{obj_pair}/{label_idx:04d}/{obj2_name}_bbox.obj"

    bbox1_pose = bbox1_p.replace(".obj",".npy")
    bbox2_pose = bbox2_p.replace(".obj",".npy")
    
    bbox1_mesh = trimesh.load(bbox1_p)
    bbox2_mesh = trimesh.load(bbox2_p)

    bbox1_pose = np.load(bbox1_pose)
    bbox2_pose = np.load(bbox2_pose)

    init_R = trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0])

    obj1_pose_inv = np.linalg.inv(obj1_pose)
    obj1_pc.apply_transform(init_R).apply_transform(obj1_pose).apply_transform(obj1_pose_inv)
    obj2_pc.apply_transform(init_R).apply_transform(obj2_pose).apply_transform(obj1_pose_inv)
    
    bbox1_mesh.apply_transform(init_R).apply_transform(bbox1_pose).apply_transform(obj1_pose_inv)
    bbox2_mesh.apply_transform(init_R).apply_transform(bbox2_pose).apply_transform(obj1_pose_inv)

    obj1_indices = points_in_bounding_box(np.array(obj1_pc.vertices), bbox1_mesh.vertices)
    obj2_indices = points_in_bounding_box(np.array(obj2_pc.vertices), bbox2_mesh.vertices)

    # Visualize the scene
    if debug:
        colors1 = np.ones((np.array(obj1_pc.vertices).shape[0], 3)) * 0
        colors1[obj1_indices, :] = [255, 0, 0]
        obj1_pc.colors = colors1
        colors2 = np.ones((np.array(obj2_pc.vertices).shape[0], 3)) * 0
        colors2[obj2_indices, :] = [0, 0, 255]
        obj2_pc.colors = colors2
        scene = trimesh.Scene()
        scene.add_geometry(obj1_pc)
        scene.add_geometry(obj2_pc)
        scene.show()

    return obj1_pc, obj2_pc, obj1_indices, obj2_indices, obj1_name, obj2_name

def color_image_based_on_position(image, bbox, alpha=128):
    """
    Similar to previous function, but use the color map instead
    """
    
    height, width = image.shape[:2]
    x0, y0, x1, y1 = map(int, bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)

    # Generate pixel coordinate grid
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    colormap_resized = colormap.resize((bbox_width, bbox_height), Image.LANCZOS)

    colors_full = np.zeros((height, width, 4), dtype=np.uint8)
    colors_full[y0:y1, x0:x1, :3] = np.array(colormap_resized)
    colors_full[y0:y1, x0:x1, 3] = alpha

    return colors_full

def bbox_from_mask(mask):
    segmentation = np.where(mask > 0.5) # be careful with binarization threshold
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))
    # [x0,y0,x1,y1], where x is width
    return [x_min, y_min, x_max, y_max]

def furthest_point_sampling(points, num_samples):
    """
    Perform furthest point sampling on a set of 2D points.
    
    Args:
        points (numpy.ndarray): An n x 2 array representing n 2D locations (pixels).
        num_samples (int): The number of samples to select.
        
    Returns:
        numpy.ndarray: Indices of the subsampled points.
    """
    n = points.shape[0]
    # Randomly initialize the first point (you can also use the first point in the array if desired)
    sampled_indices = [np.random.randint(n)]
    
    # Keep track of distances to the nearest sampled point
    distances = np.full(n, np.inf)
    
    for _ in range(1, num_samples):
        # Update distances: find the distance to the nearest sampled point
        last_sampled = points[sampled_indices[-1]]
        dist_to_last_sampled = np.linalg.norm(points - last_sampled, axis=1)
        distances = np.minimum(distances, dist_to_last_sampled)
        
        # Select the point that is farthest from the already sampled points
        # np.argmax returns first occurance!
        next_index = np.argmax(distances)
        sampled_indices.append(next_index)
        
    return np.array(sampled_indices)

def get_visible_points(extrinsic_matrix, intrinsic_matrix, points):
    # Image dimensions
    image_width = 490
    image_height = 490

    # Step 1: Convert the 3D points from world coordinates to camera coordinates
    # Add a column of 1s to the point cloud to make it homogeneous (Nx4)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])

    # Apply the extrinsic matrix to transform to camera coordinates
    points_camera = extrinsic_matrix @ points_homogeneous.T  # Shape: (4, N)
    points_camera = points_camera.T  # Shape: (N, 4)

    # Step 2: Filter points that are behind the camera (Z <= 0)
    points_camera = points_camera[points_camera[:, 2] > 0]

    # Step 3: Project the 3D points in camera coordinates to the 2D image plane
    X = points_camera[:, 0]
    Y = points_camera[:, 1]
    Z = points_camera[:, 2]

    # Apply the intrinsic matrix to project onto 2D image plane
    u = (intrinsic_matrix[0, 0] * X / Z) + intrinsic_matrix[0, 2]  # Projected x-coordinate (u)
    v = (intrinsic_matrix[1, 1] * Y / Z) + intrinsic_matrix[1, 2]  # Projected y-coordinate (v)

    # Convert u, v into integer pixel coordinates
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Step 4: Initialize a depth buffer and a visibility map to track points per pixel
    depth_buffer = np.full((image_height, image_width), np.inf)  # Infinite depth initially
    visible_mask = np.zeros(points_camera.shape[0], dtype=bool)  # Mask for visible points
    pixel_to_point_map = np.full((image_height, image_width), -1)  # Track the index of the visible point at each pixel

    # Step 5: Iterate through all points and update depth buffer
    for i in range(points_camera.shape[0]):
        if 0 <= u[i] < image_width and 0 <= v[i] < image_height:
            # If the current point is closer to the camera than the one already in the depth buffer
            if Z[i] < depth_buffer[v[i], u[i]]:
                # Unmask the previous point that was visible at this pixel
                previous_point_idx = pixel_to_point_map[v[i], u[i]]
                if previous_point_idx != -1:
                    visible_mask[previous_point_idx] = False  # Unmark the previously visible point

                # Update depth buffer and mark the new point as visible
                depth_buffer[v[i], u[i]] = Z[i]  # Update with the closer depth
                visible_mask[i] = True  # Mark this point as visible
                pixel_to_point_map[v[i], u[i]] = i  # Update the map to the new point

    # Step 6: Get indices of visible points
    visible_indices = np.where(visible_mask)[0]
    
    return visible_indices

def pick_view_source(obj_name, visible_indicies, action):
    anno = np.load(os.path.join(MESH_FULL_PATH, obj_name, "visible_pt_dict.npz"), allow_pickle=True)
    anno = {key: anno[key] for key in anno}
    overlap_count = []
    for idx in range(30):
        visible_pt = anno[str(idx)].flatten()[0]
        vis_pt_idx = visible_pt["vis_pt_idx"]
        overlap = np.intersect1d(vis_pt_idx, visible_indicies)
        overlap_count.append(len(overlap))
    top_5_indices = np.argsort(overlap_count)[-5:][::-1]
    selected_index = random.sample(list(top_5_indices), 1)[0]
    imgName = f"{selected_index:04}.png"
    return imgName

def pick_view_target(obj_name, obj_pc, obj2_indices, action, source_img_name):
    K = np.load(os.path.join(MESH_FULL_PATH, obj_name, "K", source_img_name.replace(".png", ".npy")))
    RT = np.load(os.path.join(MESH_FULL_PATH, obj_name, "RT", source_img_name.replace(".png", ".npy")))
    RT = np.vstack([RT, [0, 0, 0, 1]])
    # still need to center it
    bounding_box = obj_pc.bounding_box_oriented
    center = bounding_box.centroid
    obj_pc.apply_translation(-center)
    obj_pc = np.array(obj_pc.vertices)
    visible_indicies = get_visible_points(RT, K, obj_pc)

    anno = np.load(os.path.join(MESH_FULL_PATH, obj_name, "visible_pt_dict.npz"), allow_pickle=True)
    anno = {key: anno[key] for key in anno}
    overlap_count = []
    for idx in range(30):
        visible_pt = anno[str(idx)].flatten()[0]
        vis_pt_idx = visible_pt["vis_pt_idx"]
        overlap = np.intersect1d(vis_pt_idx, visible_indicies)
        overlap2 = np.intersect1d(vis_pt_idx, obj2_indices)
        # combine both measures, we want the pose to align and 
        # we want the part to be as visible as possible
        score = 0.5*(len(overlap) / len(visible_indicies)) + 0.5*(len(overlap2) / len(obj2_indices))
        overlap_count.append(score)
    
    # pick a random view:
    top_5_indices = np.argsort(overlap_count)[-5:][::-1]
    selected_index = random.sample(list(top_5_indices), 1)[0]
    imgName = f"{selected_index:04}.png"

    return imgName

def create_square_bbox(mask, pad_ratio = 1.2):
    mask_array = np.array(mask) // 255
    segmentation = np.where(mask_array > 0.5) # be careful with binarization threshold
    x0 = int(np.min(segmentation[1]))
    x1 = int(np.max(segmentation[1]))
    y0 = int(np.min(segmentation[0]))
    y1 = int(np.max(segmentation[0]))
    # [x0,y0,x1,y1], where x is width

    # Calculate the center of the original bounding box
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    
    # Calculate the width and height of the original bounding box
    width = x1 - x0
    height = y1 - y0
    
    # Find the longest side and compute the side length of the new square bounding box
    max_side = max(width, height)
    new_side = pad_ratio * max_side
    
    # Calculate the new bounding box coordinates
    new_x0 = center_x - new_side // 2
    new_y0 = center_y - new_side // 2
    new_x1 = center_x + new_side // 2
    new_y1 = center_y + new_side // 2

    new_x0 = int(new_x0)
    new_y0 = int(new_y0)
    new_x1 = int(new_x1)
    new_y1 = int(new_y1)
    
    return [new_x0, new_y0, new_x1, new_y1]

def create_mask_from_label(obj_name, img_name, visible_indicies):
    anno = np.load(os.path.join(MESH_FULL_PATH, obj_name, "visible_pt_dict.npz"), allow_pickle=True)
    anno = {key: anno[key] for key in anno}
    visible_pt = anno[str(int(img_name.replace(".png", "")))].flatten()[0]
    vis_pt_idx = visible_pt["vis_pt_idx"]
    proj_2D_loc = visible_pt["proj_2D_loc"]
    overlap = np.intersect1d(vis_pt_idx, visible_indicies)
    indices_in_overlap = np.where(np.isin(vis_pt_idx, overlap))[0]
    vis_pt_idx = vis_pt_idx[indices_in_overlap]
    proj_2D_loc = proj_2D_loc[indices_in_overlap]

    object_mask = Image.open(f"{MESH_FULL_PATH}/{obj_name}/segmentation/{img_name.replace('.png', '_0001.png')}").convert("L")
    # pad it to larger image
    padding = (
        60,  # left padding
        60   # top padding
    )
    object_mask = ImageOps.expand(object_mask, border=padding, fill=0)
    crop_bbox = create_square_bbox(object_mask)
    object_mask = object_mask.crop(crop_bbox)
    object_mask = object_mask.resize((224, 224), Image.NEAREST)

    # process point coordinates
    proj_2D_loc[:, 0] += 60
    proj_2D_loc[:, 1] += 60
    proj_2D_loc[:, 0] -= crop_bbox[1]
    proj_2D_loc[:, 1] -= crop_bbox[0]
    ratio = 224 / (crop_bbox[2] - crop_bbox[0])
    proj_2D_loc = np.round(proj_2D_loc * ratio).astype(int)
    proj_2D_loc[proj_2D_loc >= 224] = 223
    proj_2D_loc[proj_2D_loc < 0] = 0

    img_height, img_width = 224, 224  # HACK: hard-coded
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    mask[proj_2D_loc[:, 0], proj_2D_loc[:, 1]] = 255

    kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=2)

    object_mask = np.array(object_mask) // 255
    mask = mask * object_mask
    mask = mask // 255
    return mask, vis_pt_idx, proj_2D_loc

def pick_best_views(action, obj1_pc, obj2_pc, obj1_indices, obj2_indices, obj1_name, obj2_name, seen_views):
    while True:
        selected_img1 = pick_view_source(obj1_name, obj1_indices, action)
        selected_img2 = pick_view_target(obj2_name, obj2_pc, obj2_indices, action, selected_img1)
        if (selected_img1, selected_img2) not in seen_views:
            seen_views.add((selected_img1, selected_img2))
            break

    # create the part masks
    part_mask1, vis_pt_idx1, proj_2D_loc1 = create_mask_from_label(obj1_name, selected_img1, obj1_indices)
    part_mask2, vis_pt_idx2, proj_2D_loc2 = create_mask_from_label(obj2_name, selected_img2, obj2_indices)

    return selected_img1, part_mask1, vis_pt_idx1, proj_2D_loc1, selected_img2, part_mask2, vis_pt_idx2, proj_2D_loc2, seen_views


def label_pointcloud(action, obj_pair, label_idx, out_dir, debug=True):
    seen_views = set()

    for trial in range(N_TRIALS):
        obj1_pc, obj2_pc, obj1_indices, obj2_indices, obj1_name, obj2_name = load_meshes_obj(action, obj_pair, label_idx)
        selected_img1, mask1, vis_pt_idx_1, proj_2D_loc1, selected_img2, mask2, vis_pt_idx_2, proj_2D_loc2, seen_views = pick_best_views(action, obj1_pc, obj2_pc, obj1_indices, obj2_indices, obj1_name, obj2_name, seen_views)
        
        # save the mask images
        mask_image = Image.fromarray(mask1 * 255)
        mask_image.save(os.path.join(MASK_OUT_DIR, f"{action}|||{obj1_name}|||{selected_img1.replace('.png', '')}|||trial_{trial}.png"))
        mask_image = Image.fromarray(mask2 * 255)
        mask_image.save(os.path.join(MASK_OUT_DIR, f"{action}|||{obj2_name}|||{selected_img2.replace('.png', '')}|||trial_{trial}.png"))

        # the 2d locations need to be unique!
        _, unique_indices1 = np.unique(proj_2D_loc1, axis=0, return_index=True)
        vis_pt_idx_1 = vis_pt_idx_1[unique_indices1]
        proj_2D_loc1 = proj_2D_loc1[unique_indices1]
        _, unique_indices2 = np.unique(proj_2D_loc2, axis=0, return_index=True)
        vis_pt_idx_2 = vis_pt_idx_2[unique_indices2]
        proj_2D_loc2 = proj_2D_loc2[unique_indices2]
        
        # Randomly permute for FPS to work correctly!
        n1 = len(vis_pt_idx_1)
        permutation_indices1 = np.random.permutation(n1)
        vis_pt_idx_1 = vis_pt_idx_1[permutation_indices1]
        proj_2D_loc1 = proj_2D_loc1[permutation_indices1]
        n2 = len(vis_pt_idx_2)
        permutation_indices2 = np.random.permutation(n2)
        vis_pt_idx_2 = vis_pt_idx_2[permutation_indices2]
        proj_2D_loc2 = proj_2D_loc2[permutation_indices2]
        
        num_points = np.min((10000, len(proj_2D_loc1), len(proj_2D_loc2))) # 10000 point limit
        if len(proj_2D_loc1) > num_points:
            selected_indices = furthest_point_sampling(proj_2D_loc1, num_points)
            vis_pt_idx_1 = vis_pt_idx_1[selected_indices]
            proj_2D_loc1 = proj_2D_loc1[selected_indices]
        if len(proj_2D_loc2) > num_points:
            selected_indices = furthest_point_sampling(proj_2D_loc2, num_points)
            vis_pt_idx_2 = vis_pt_idx_2[selected_indices]
            proj_2D_loc2 = proj_2D_loc2[selected_indices]

        obj1_pc = obj1_pc[vis_pt_idx_1]
        obj2_pc = obj2_pc[vis_pt_idx_2]

        distances = distance.cdist(obj1_pc, obj2_pc, metric='euclidean')
        start_time = time.time()
        row_ind, col_ind = linear_sum_assignment(distances)
        print(time.time() -  start_time)
        matched_in_2 = col_ind

        # save the matches:
        os.makedirs(f"{out_dir}/trial_{trial}", exist_ok=True)
        np.save(f"{out_dir}/trial_{trial}/{obj1_name}|||{selected_img1}.npy", proj_2D_loc1)
        np.save(f"{out_dir}/trial_{trial}/{obj2_name}|||{selected_img2}.npy", proj_2D_loc2[matched_in_2])

        # visualize
        image1 = Image.open(f"{RESIZED_IMAGE_PATH}/{obj1_name}/rgb_images_processed/{selected_img1}").convert("RGBA")
        image2 = Image.open(f"{RESIZED_IMAGE_PATH}/{obj2_name}/rgb_images_processed/{selected_img2}").convert("RGBA")
        
        bbox = bbox_from_mask(mask1)
        # use the colormap directory instead
        colors1 = color_image_based_on_position(np.array(image1), bbox=bbox, alpha=180)
        colors1 *= np.expand_dims(mask1, axis=-1)
        color_map_image1 = Image.fromarray(colors1, 'RGBA')
        blended_image1 = Image.alpha_composite(image1, color_map_image1)
        
        height, width = np.array(image2).shape[:2]
        colors2 = np.zeros((height, width, 4), dtype=np.uint8)
        for i, pixel1 in enumerate(proj_2D_loc1): 
            match = matched_in_2[i]
            pixel2 = proj_2D_loc2[match]
            colors2[pixel2[0], pixel2[1], :] = colors1[pixel1[0], pixel1[1], :]

        # NN interpolation
        colored_coords = np.argwhere(colors2[...,3] == 180)
        colored_colors = colors2[colors2[...,3] == 180]
        uncolored_coords = np.argwhere(colors2[...,3] != 180)
        kdtree = KDTree(colored_coords)
        for coord in uncolored_coords:
            dist, nearest_idx = kdtree.query(coord)
            # Assign the color of the nearest colored pixel
            if dist < 2:
                colors2[tuple(coord)] = colored_colors[nearest_idx]

        color_map_image2 = Image.fromarray(colors2, 'RGBA')
        blended_image2 = Image.alpha_composite(image2, color_map_image2)
        combined_img = Image.new('RGBA', (width * 2, height))
        combined_img.paste(blended_image1, (0, 0))
        combined_img.paste(blended_image2, (width, 0))
        combined_img.save(os.path.join(VIZ_OUT_DIR, f"{action}|||{obj1_name}|||{selected_img1.replace('.png', '')}|||{obj2_name}|||{selected_img2.replace('.png', '')}|||trial_{trial}.png"))
        
def main(actions, cluster_id):
    time.sleep(cluster_id*1)

    for action in actions:
        print(f"Processing action: {action}")
        obj_pairs = sorted(os.listdir(f"{ANNO_PATH}/{action}"))
        obj_pairs = [x for x in obj_pairs if x[0]!="."]

        for i, obj_pair in enumerate(obj_pairs):
            pair_path = f"{ANNO_PATH}/{action}/{obj_pair}"
            temp = os.listdir(pair_path)
            temp = [x for x in temp if x[0]!="."]
            n_labels = len(temp)

            for label_idx in range(n_labels):
                print(f"Rendering {action} | {obj_pair} | {label_idx:04d}")
                item_out_path = f"{OUT_DIR}/{action}/{obj_pair}|||{label_idx:04d}"
                os.makedirs(item_out_path, exist_ok=True)
                try:
                    label_pointcloud(action, obj_pair, label_idx, item_out_path)
                except Exception as e:
                    # Print the traceback of the exception
                    print(f"Error: {e}")
                    traceback.print_exc()

def split_list(lst, n):
    avg_len = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

if __name__ == "__main__":
    actions = sorted(os.listdir(ANNO_PATH))
    actions = [x for x in actions if x[0]!="."]
    num_cpus = 24
    actions = split_list(actions, num_cpus)
    with ProcessPoolExecutor(max_workers=num_cpus) as main_executor:
        main_futures = [main_executor.submit(main, actions[i], i) for i in range(num_cpus)]
    