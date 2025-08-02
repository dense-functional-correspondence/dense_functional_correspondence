import os
import sys
import numpy as np
import json
import glob
import cv2
from PIL import Image
import random
from cond_corr.data.utils.config_gen_utils import colors, get_aff_json_name, load_mask, draw_bounding_box, check_valid_bbox_anno
from concurrent.futures import ProcessPoolExecutor

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

def find_spatial_correspondence(vis_pt_idx1, proj_2D_loc1, vis_pt_idx2, proj_2D_loc2, mask1, mask2):
    # Find the common elements and their indices
    common_elements = np.intersect1d(vis_pt_idx1, vis_pt_idx2)
    indices1 = np.where(np.isin(vis_pt_idx1, common_elements))[0]
    indices2 = np.where(np.isin(vis_pt_idx2, common_elements))[0]
    points1 = proj_2D_loc1[indices1]
    points2 = proj_2D_loc2[indices2]

    # get rid of duplicate pairs
    valid_mask1 = mask1[points1[:, 0], points1[:, 1]] == 1
    valid_mask2 = mask2[points2[:, 0], points2[:, 1]] == 1
    combined_valid_mask = valid_mask1 & valid_mask2
    points1 = points1[combined_valid_mask]
    points2 = points2[combined_valid_mask]

    # correspondence should be a well-defined function
    _, unique_indices = np.unique(points1, axis=0, return_index=True)
    points1 = points1[unique_indices]
    points2 = points2[unique_indices]

    assert(len(points1) == len(points2))
    return points1, points2

def run_main(assets, cluster_id, dataset_path):
    collected_annotations = []
    num_objects = 0

    for asset_idx, asset in enumerate(assets):
        num_objects += 1
        img_dir = os.path.join(dataset_path, "fully_processed_data", asset, "rgb_images_processed")
        anno_dir = os.path.join(dataset_path, "fully_processed_data", asset, "visible_pt_dict_processed.npz")
        label_dir = os.path.join(dataset_path, "fully_processed_data", asset, "pseudo_labels_processed")
        assert os.path.exists(img_dir)
        assert os.path.exists(anno_dir)
        assert os.path.exists(label_dir)
        visible_pt_dict = np.load(anno_dir, allow_pickle=True)
        visible_pt_dict = {key: visible_pt_dict[key] for key in visible_pt_dict}
        images = os.listdir(img_dir)
        random.shuffle(images) # so that it's not always in the same order...
        parts = os.listdir(label_dir)

        vis_pt_idx_list = []
        proj_2D_loc_list = []
        label_list = {}
        for part in parts:
            label_list[part] = []

        for image in images:
            img_num = image
            visible_pt = visible_pt_dict[img_num].flatten()[0]
            vis_pt_idx = visible_pt["vis_pt_idx"]
            vis_pt_idx_list.append(vis_pt_idx)
            proj_2D_loc = visible_pt["proj_2D_loc"]
            proj_2D_loc_list.append(proj_2D_loc)

            for part in parts:
                label = Image.open(os.path.join(label_dir, part, image)).convert('L')
                label = np.array(label) // 255
                label_list[part].append(label)

        for i in range(len(images)):
            image1 = images[i]
            vis_pt_idx1 = vis_pt_idx_list[i]
            proj_2D_loc1 = proj_2D_loc_list[i]
            for j in range(i+1, len(images)):
                image2 = images[j]
                vis_pt_idx2 = vis_pt_idx_list[j]
                proj_2D_loc2 = proj_2D_loc_list[j]

                valid_parts = []
                for part in parts:
                    points1, points2 = find_spatial_correspondence(vis_pt_idx1, proj_2D_loc1, vis_pt_idx2, proj_2D_loc2, label_list[part][i], label_list[part][j])
                    if len(points1) >= 128:  # minimum number of required common points
                        valid_parts.append(part)
                
                if len(valid_parts) > 0:
                    collected_annotations.append([asset, image1, image2, valid_parts])                     
                    print(f"{asset_idx}/{len(assets)}: ", len(collected_annotations), end='\r')
    return num_objects, collected_annotations

def main():
    dataset_path =  "<#TODO: path to the dataset root>"
    metadata_path = "../0_taxonomy_and_metadata/Objaverse/"
    
    for split in ["train", "test", "unseen_test"]:
        assets = json.load(open(os.path.join(metadata_path, f"{split}_split.json")))
        
        num_cpus = 16
        print(f"Using {num_cpus} CPUs.")
        assets = split_list(assets, num_cpus)
        with ProcessPoolExecutor(max_workers=num_cpus) as main_executor:
            main_futures = [main_executor.submit(run_main, assets[i], i, dataset_path) for i in range(num_cpus)]
        all_annotations = []
        total_num_objects = 0
        for future in main_futures:
            num_objects, collected_annotations = future.result()
            all_annotations += collected_annotations
            total_num_objects += num_objects                  

        print(f"On the {split} split, gathered {len(all_annotations)} data points from {total_num_objects} objects.")
        
        with open(os.path.join(metadata_path, f"{split}_spatial_part_list.json"), "w") as f:
            f.write(json.dumps(all_annotations, indent=True))
     
if __name__ == "__main__":
    main()