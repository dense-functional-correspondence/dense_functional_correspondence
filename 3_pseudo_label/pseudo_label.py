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
import cv2
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    normalized_array = (x - min_val) / (max_val - min_val)
    return normalized_array

def main(category, part_dir, crop_dict=None, edge_dict=None):
    visible_pt_dict = np.load(os.path.join(part_dir, "visible_pt_dict.npz"), allow_pickle=True)
    visible_pt_dict = {key: visible_pt_dict[key] for key in visible_pt_dict}
    label_dir = part_dir.replace("renders_for_training_complete", "renders_for_labeling_complete")
    if os.path.exists(os.path.join(label_dir, "heatmap_aggregated")):
        heatmap_dir = os.path.join(label_dir, "heatmap_aggregated")
    else:
        # some categories may not be processed / used.
        return

    affordance_annos = [name for name in os.listdir(heatmap_dir)]
    # no affordance found
    if len(affordance_annos) == 0:
        return
    os.makedirs(os.path.join(part_dir, "pseudo_labels"), exist_ok=True)

    for i, affordance_anno in enumerate(affordance_annos):
        affordance = affordance_anno.split("|||")[1]
        os.makedirs(os.path.join(part_dir, "pseudo_labels", affordance), exist_ok=True)

        heatmap = np.load(os.path.join(heatmap_dir, affordance_anno))
        heatmap = normalize(heatmap)

        for img_name in os.listdir(os.path.join(part_dir, "rgb_images")):
            mask = Image.open(os.path.join(part_dir, "segmentation", img_name.replace(".png", "_0001.png"))).convert('L')
            mask = np.array(mask) // 255
            img_num = str(int(img_name.removesuffix('.png')))
            visible_pt = visible_pt_dict[img_num].flatten()[0]
            vis_pt_idx = visible_pt["vis_pt_idx"]
            proj_2D_loc = visible_pt["proj_2D_loc"]
            valid_indices = np.where((proj_2D_loc[:, 0] >= 0) & (proj_2D_loc[:, 0] <= 489) & (proj_2D_loc[:, 1] >= 0) & (proj_2D_loc[:, 1] <= 489))
            vis_pt_idx = vis_pt_idx[valid_indices]
            proj_2D_loc = proj_2D_loc[valid_indices]
            indexed_heatmap = heatmap[vis_pt_idx]

            output_array = np.zeros((490, 490))
            output_array[proj_2D_loc[:, 0], proj_2D_loc[:, 1]] = indexed_heatmap
            image_array = (output_array * 255).astype(np.uint8)
            
            if len(np.where(indexed_heatmap > 0.5)[0]) < 200: 
                # not many visible highlighted points
                # for example, if the view is top-down but the part is base
                # the heatmap would be random but likely smaller than 0.5.
                image_array[image_array < 128] = 0
                image_array[image_array >= 128] = 255
            else:
                threshold1 = 64
                image_array[image_array < threshold1] = threshold1
                if edge_dict is not None and edge_dict[affordance.replace('_', ' ')]: # use edge, thin!
                    blur = cv2.GaussianBlur(image_array,(3,3),0)
                    threshold2,image_array = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                else:
                    blur = cv2.GaussianBlur(image_array,(11,11),0)
                    threshold2,image_array = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                if threshold2 < threshold1:
                    image_array = (output_array * 255).astype(np.uint8)
                    image_array[image_array < 128] = 0
                    image_array[image_array >= 128] = 255
            
            # closing and openning
            if crop_dict is not None and crop_dict[affordance.replace('_', ' ')]: # small parts and edges
                kernel = np.ones((3, 3), np.uint8)
                image_array = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel, iterations=3)
            elif edge_dict is not None and edge_dict[affordance.replace('_', ' ')]: # small parts and edges
                kernel = np.ones((3, 3), np.uint8)
                image_array = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel, iterations=3)
            else:
                kernel = np.ones((5, 5), np.uint8) # larger parts
                image_array = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel, iterations=5)

            # multiply by mask
            image_array = image_array * mask
            image = Image.fromarray(image_array)
            image.save(os.path.join(part_dir, "pseudo_labels", affordance, img_name))

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

def run_main(categories, cluster_id, render_dir, crop_requirement, edge_requirement):
    time.sleep(cluster_id*1) # hopefully avoid different processes writing at the same time...
    start_time = time.time()
    for cat_idx, category in enumerate(categories):
        print(f"\nCluster{cluster_id}-{cat_idx+1}/{len(categories)}: Running pseudo-labelling on {category} images!")

        part_dir = os.path.join(render_dir, category)
        base_category = category.split("---")[-1].replace(".glb","") 

        if crop_requirement is not None:
            crop_dict = crop_requirement[base_category]
        else:
            crop_dict = None
        
        if edge_requirement is not None:
            edge_dict = edge_requirement[base_category]
        else:
            edge_dict = None

        main(category, part_dir, crop_dict, edge_dict)

        print(f"Running time: {time.time() - start_time} seconds")
    time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('category', help='help')
    parser.add_argument('--render_dir', type=str, help="<#TODO: path to the rendered data directory>")
    parser.add_argument('--crop_requirement_dir', type=str, help="<#TODO: path to crop_requirement.json>")
    parser.add_argument('--edge_requirement_dir', type=str, help="<#TODO: path to edge_requirement.json>")
    parser.add_argument('--all_category', action='store_true', help='run on all categories')
    args = parser.parse_args()
    
    if args.edge_requirement_dir is not None:
        edge_requirement = json.load(open(args.edge_requirement_dir))
    else:
        edge_requirement = None
    if args.crop_requirement_dir is not None:
        crop_requirement = json.load(open(args.crop_requirement_dir))
    else:
        crop_requirement = None

    render_dir = args.render_dir
    if args.all_category:
        categories = [name for name in os.listdir(render_dir) if os.path.isdir(os.path.join(render_dir, name))]
    else:
        categories = args.category.split(",")
        
    num_cpus = 4
    print(f"Using {num_cpus} CPUs.")
    random.shuffle(categories)
    categories = split_list(categories, num_cpus)
    print(categories)

    with ProcessPoolExecutor(max_workers=num_cpus) as main_executor:
        main_futures = [main_executor.submit(run_main, categories[i], i, render_dir, crop_requirement, edge_requirement) for i in range(num_cpus)]

        for i, future in enumerate(as_completed(main_futures)):
            try:
                result = future.result()
                print(f"Cluster {i+1} finished with result: {result}")
            except Exception as e:
                print(f"Cluster {i+1} raised an exception: {e}")