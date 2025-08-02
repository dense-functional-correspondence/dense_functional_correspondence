import os
import shutil
import time
import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import KDTree

def is_within_bounding_box(pixel_location, bounding_box):
    """
    Check if a 2D pixel location is within a bounding box.

    Parameters:
    - pixel_location (tuple): A tuple (x, y) representing the pixel location.
    - bounding_box (list or tuple): A list or tuple [x0, y0, x1, y1] representing the bounding box.

    Returns:
    - bool: True if the pixel location is within the bounding box, False otherwise.
    """
    x, y = pixel_location
    x0, y0, x1, y1 = bounding_box

    return x0 <= y <= x1 and y0 <= x <= y1

def count_within_bounding_boxes(proj_2D_loc, bboxes):
    """
    Count how many bounding boxes each 2D pixel location is within.

    Parameters:
    - proj_2D_loc (numpy.ndarray): An array of shape (N, 2) representing the 2D pixel locations.
    - bboxes (numpy.ndarray): An array of shape (M, 4) representing the bounding boxes [x0, y0, x1, y1].

    Returns:
    - numpy.ndarray: An array of shape (N,) where each element is the count of bounding boxes the pixel location is within.
    """
    x = proj_2D_loc[:, 0:1]  # Shape (N, 1)
    y = proj_2D_loc[:, 1:2]  # Shape (N, 1)
    
    x0 = bboxes[:, 0]  # Shape (M,)
    y0 = bboxes[:, 1]  # Shape (M,)
    x1 = bboxes[:, 2]  # Shape (M,)
    y1 = bboxes[:, 3]  # Shape (M,)
    
    in_bbox = (y >= x0) & (y <= x1) & (x >= y0) & (x <= y1)  # Shape (N, M)
    counts = np.sum(in_bbox, axis=1)
    return counts

def main(category, part_dir, points, crop_dict = None, edge_dict = None):
    visible_pt_dict = np.load(os.path.join(part_dir, "visible_pt_dict.npz"), allow_pickle=True)
    visible_pt_dict = {key: visible_pt_dict[key] for key in visible_pt_dict}
    if os.path.exists(os.path.join(part_dir, "part_annotations")):
        bbox_dir = os.path.join(part_dir, "part_annotations")
    else:
        # some categories may not be processed / used.
        return

    affordance_annos = [name for name in os.listdir(bbox_dir) if "aggregated" not in name]
    # no affordance found
    if len(affordance_annos) == 0:
        return
    os.makedirs(os.path.join(part_dir, "heatmap_3dviz"), exist_ok=True)
    os.makedirs(os.path.join(part_dir, "heatmap_aggregated"), exist_ok=True)

    # preprocess the edges:
    edge_dir = os.path.join(part_dir, "primitives", "results", "edge.txt")
    if os.path.exists(edge_dir):
        edge_prob = np.loadtxt(edge_dir, delimiter=';')
        edge_prob = edge_prob[:, 1]
        edge_ptcloud_dir = os.path.join(part_dir, "primitives", "sampled_point_cloud.npy")
        edge_ptcloud = np.load(edge_ptcloud_dir)
        kd_tree = KDTree(edge_ptcloud)
        distances, indices = kd_tree.query(points)
        edges = edge_prob[indices]
    else:
        edges = None

    for i, affordance_anno in enumerate(affordance_annos):
        affordance = affordance_anno.removesuffix('.json')

        files = os.listdir(os.path.join(part_dir, "heatmap_aggregated"))
        for file in files:
            if f"{category}|||{affordance}" in file:
                os.remove(os.path.join(part_dir, "heatmap_aggregated", file))
        files = os.listdir(os.path.join(part_dir, "heatmap_3dviz"))
        for file in files:
            if f"{category}|||{affordance}" in file:
                os.remove(os.path.join(part_dir, "heatmap_3dviz", file))

        if crop_dict is not None and crop_dict[affordance.replace('_', ' ')]:
            zoom_in = True
        else:
            zoom_in = False

        if edge_dict is not None and edge_dict[affordance.replace('_', ' ')]:
            use_edge = True
            assert(edges is not None)
        else:
            use_edge = False
        
        if zoom_in:
            with open(os.path.join(part_dir, "part_annotations_cropped", affordance_anno)) as f:
                img_to_bbox = json.load(f)
        else:
            with open(os.path.join(bbox_dir, affordance_anno)) as f:
                img_to_bbox = json.load(f)
        if zoom_in:
            crop_offset = json.load(open(os.path.join(part_dir, "crop_offset", f"{affordance}_crop_offset.json")))

        aggregated_heatmap = np.zeros(points.shape[0])

        for img_name, info in img_to_bbox.items():
            img_num = str(int(img_name.removesuffix('.png')))
            visible_pt = visible_pt_dict[img_num].flatten()[0]
            bboxes = list(info.values())
            assert(len(bboxes) == 1)
            bboxes = np.array(bboxes[0])
            if len(bboxes) <= 2: # if less than 2 valid boxes
                continue
            if zoom_in:
                bboxes[:, 0] += crop_offset[img_name][0]
                bboxes[:, 1] += crop_offset[img_name][1]
                bboxes[:, 2] += crop_offset[img_name][0]
                bboxes[:, 3] += crop_offset[img_name][1]

            vis_pt_idx = visible_pt["vis_pt_idx"]
            proj_2D_loc = visible_pt["proj_2D_loc"]
            counts = count_within_bounding_boxes(proj_2D_loc, bboxes)
            
            np.add.at(aggregated_heatmap, vis_pt_idx, counts)
        
        if use_edge:
            aggregated_heatmap *= edges # multiply by edge probability

        suffix = ""
        if zoom_in:
            suffix += "-cropped"
        if use_edge:
            suffix += "-with_edge"

        np.save(os.path.join(part_dir, "heatmap_aggregated", f"{category}|||{affordance}|||3Dheatmap{suffix}.npy"), aggregated_heatmap)
        # visualize it
        aggregated_heatmap = np.load(os.path.join(part_dir, "heatmap_aggregated", f"{category}|||{affordance}|||3Dheatmap{suffix}.npy"))
        min_val = np.min(aggregated_heatmap)
        max_val = np.max(aggregated_heatmap)
        normalized_array = (aggregated_heatmap - min_val) / (max_val - min_val)
        colormap = plt.get_cmap('viridis') # coolwarm, inferno, viridis
        colors_rgb = (colormap(normalized_array)[:, :3] * 255).astype(np.uint8)
        # Create a Trimesh PointCloud
        point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors_rgb)
        # Save the point cloud to a PLY file
        point_cloud.export(os.path.join(part_dir, "heatmap_3dviz", f"{category}|||{affordance}|||ptcloud{suffix}.ply"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('category', help='help')
    parser.add_argument('--render_dir', type=str, help="<#TODO: path to the folder that contains the rendered images>")
    parser.add_argument('--asset_dir', type=str, help="<#TODO: path to the folder that contains the point clouds>")
    parser.add_argument('--edge_requirement_dir', type=str, help="<#TODO: path to edge_requirement.json>")
    parser.add_argument('--crop_requirement_dir', type=str, help="<#TODO: path to crop_requirement.json>")
    parser.add_argument('--all_category', action='store_true', help='run on all categories')
    args = parser.parse_args()

    render_dir = args.render_dir
    asset_dir = args.asset_dir

    if args.edge_requirement_dir is not None:
        edge_requirement = json.load(open(args.edge_requirement_dir))
    else:
        edge_requirement = None
    if args.crop_requirement_dir is not None:
        crop_requirement = json.load(open(args.crop_requirement_dir))
    else:
        crop_requirement = None
    
    if args.all_category:
        categories = [name for name in os.listdir(render_dir) if os.path.isdir(os.path.join(render_dir, name))]
    else:
        categories = args.category.split(",")

    start_time = time.time()
    for cat_idx, category in enumerate(categories):
        print(f"\n{cat_idx+1}/{len(categories)}: Running visualizing 3D heatmap on {category} images!")

        part_dir = os.path.join(render_dir, category)
        uid = category.split("---")[0]
        point_cloud_dir = os.path.join(asset_dir, f"{uid}.npy")
        point_cloud = np.load(point_cloud_dir)[:, :3]
        base_category = category.split("---")[-1].replace(".glb","")
        
        if crop_requirement is not None:
            crop_dict = crop_requirement[base_category]
        else:
            crop_dict = None
        
        if edge_requirement is not None:
            edge_dict = edge_requirement[base_category]
        else:
            edge_dict = None

        main(category, part_dir, point_cloud, crop_dict=crop_dict, edge_dict=edge_dict)

        print(f"Running time: {time.time() - start_time} seconds")