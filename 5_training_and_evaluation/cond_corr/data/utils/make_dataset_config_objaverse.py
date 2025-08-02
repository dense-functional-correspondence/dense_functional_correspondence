import os
import numpy as np
import json
import glob
import cv2
from PIL import Image
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

def run_main(assets, cluster_id, dataset_path, taxonomy, duplicated_parts):
    collected_annotations = []
    num_objects = 0

    for asset_idx, asset in enumerate(assets):
        num_objects += 1
        asset_category = asset.split("---")[-1].replace(".glb", "")
        affordance2parts = taxonomy[asset_category]

        img_dir = os.path.join(dataset_path, "fully_processed_data", asset, "rgb_images_processed")
        mask_dir = os.path.join(dataset_path, "fully_processed_data", asset, "object_masks_processed")
        part_anno_dir = os.path.join(dataset_path, "fully_processed_data", asset, "pseudo_labels_processed")
        assert os.path.exists(img_dir)
        assert os.path.exists(mask_dir)
        assert os.path.exists(part_anno_dir)

        for affordance, parts in affordance2parts.items():
            images = os.listdir(img_dir)
            for image in images:
                bbox_idxs = []
                for part in parts:
                    part_name = part[2:].strip().replace('*', '')
                    part_name = part_name.split("- ")[0].strip().lower()
                    if part_name in duplicated_parts[asset_category]:
                        dedup_part_name = duplicated_parts[asset_category][part_name]
                    else:
                        dedup_part_name = part_name
                    dedup_part_name = dedup_part_name.replace(" ", "_")
                    part_pseudo_labels_p = os.path.join(part_anno_dir, dedup_part_name)

                    pseudo_label_p = os.path.join(part_pseudo_labels_p, image)
                    pseudo_label = Image.open(pseudo_label_p).convert("L")
                    image_array = np.array(pseudo_label)
                    if np.sum(image_array // 255) > 32:  # use 32 as threshold
                        bbox_idxs.append(dedup_part_name)

                if len(bbox_idxs) > 0:
                    collected_annotations.append([asset, image, affordance, bbox_idxs])                     

                print(f"{asset_idx}/{len(assets)}: ", len(collected_annotations), end='\r')    
    return num_objects, collected_annotations

def main():
    dataset_path =  "<#TODO: path to the dataset root>"
    metadata_path = "../0_taxonomy_and_metadata/Objaverse/"
    taxonomy_p = os.path.join(metadata_path, "obj2obj_fewshot_manually_processed.json")
    taxonomy = json.load(open(taxonomy_p))
    duplicated_parts_p = os.path.join(metadata_path, "duplicated_parts.json")
    duplicated_parts = json.load(open(duplicated_parts_p))

    for split in ["train", "test", "unseen_test"]: 
        assets = json.load(open(os.path.join(metadata_path, f"{split}_split.json")))
        num_cpus = 8
        print(f"Using {num_cpus} CPUs.")
        assets = split_list(assets, num_cpus)
        with ProcessPoolExecutor(max_workers=num_cpus) as main_executor:
            main_futures = [main_executor.submit(run_main, assets[i], i, dataset_path, taxonomy, duplicated_parts) for i in range(num_cpus)]
        all_annotations = []
        total_num_objects = 0
        for future in main_futures:
            num_objects, collected_annotations = future.result()
            all_annotations += collected_annotations
            total_num_objects += num_objects

        print(f"On the {split} split, gathered {len(all_annotations)} data points from {total_num_objects} objects.")
        
        with open(os.path.join(metadata_path, f"{split}_list.json"), "w") as f:
            f.write(json.dumps(all_annotations, indent=True))
     
if __name__ == "__main__":
    main()