import os
import time
import json
import argparse
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
from concurrent.futures import ProcessPoolExecutor
import random

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

def run_main(args, categories, cluster_id):
    time.sleep(cluster_id*1) # hopefully avoid different processes writing at the same time...
    
    start_time = time.time()
    for cat_idx, category in enumerate(categories):
        print(f"\nCluster{cluster_id}-{cat_idx+1}/{len(categories)}: Running crop-padding on {category} images!")

        obj_dir = os.path.join(args.root_dir, category)
        old_img_dir = os.path.join(obj_dir, "rgb_images")
        old_mask_dir = os.path.join(obj_dir, "segmentation")
        old_label_dir = os.path.join(obj_dir, "pseudo_labels")

        out_dir = os.path.join(args.out_dir, category)
        if os.path.exists(out_dir): # this one is done!
            continue
        
        new_img_dir = os.path.join(out_dir, "rgb_images_processed")
        new_mask_dir = os.path.join(out_dir, "object_masks_processed")
        new_label_dir = os.path.join(out_dir, "pseudo_labels_processed")
        os.makedirs(new_img_dir, exist_ok = True)
        os.makedirs(new_mask_dir, exist_ok = True)
        os.makedirs(new_label_dir, exist_ok = True)

        for part in os.listdir(old_label_dir):
            os.makedirs(os.path.join(new_label_dir, part), exist_ok = True)
        
        # visible pt dict processing:
        visible_pt_dict = np.load(os.path.join(obj_dir, "visible_pt_dict.npz"), allow_pickle=True)
        visible_pt_dict = {key: visible_pt_dict[key] for key in visible_pt_dict}
        visible_pt_dict_processed = {}

        for image_name in os.listdir(old_img_dir):
            image = Image.open(os.path.join(old_img_dir, image_name)).convert('RGB')
            mask = Image.open(os.path.join(old_mask_dir, image_name.replace(".png", "_0001.png"))).convert('L')
            white_background = Image.new('RGB', image.size, (255, 255, 255))
            image = Image.composite(image, white_background, mask)

            img_num = str(int(image_name.removesuffix('.png')))
            visible_pt = visible_pt_dict[img_num].flatten()[0]
            visible_pt_dict_processed[image_name] = {"vis_pt_idx": None, "proj_2D_loc": None}
            vis_pt_idx = visible_pt["vis_pt_idx"]
            proj_2D_loc = visible_pt["proj_2D_loc"]
            valid_indices = np.where((proj_2D_loc[:, 0] >= 0) & (proj_2D_loc[:, 0] <= 489) & (proj_2D_loc[:, 1] >= 0) & (proj_2D_loc[:, 1] <= 489))
            vis_pt_idx = vis_pt_idx[valid_indices]
            proj_2D_loc = proj_2D_loc[valid_indices]

            # pad it to larger image
            padding = (
                60,  # left padding
                60   # top padding
            )
            image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))
            mask = ImageOps.expand(mask, border=padding, fill=0)

            crop_bbox = create_square_bbox(mask)
            image = image.crop(crop_bbox)
            image = image.resize((224, 224), Image.LANCZOS)
            image.save(os.path.join(new_img_dir, image_name))
            mask = mask.crop(crop_bbox)
            mask = mask.resize((224, 224), Image.NEAREST)
            mask.save(os.path.join(new_mask_dir, image_name))

            mask = np.array(mask) // 255

            # process visible points coordinates
            proj_2D_loc[:, 0] += 60
            proj_2D_loc[:, 1] += 60
            proj_2D_loc[:, 0] -= crop_bbox[1]
            proj_2D_loc[:, 1] -= crop_bbox[0]
            ratio = 224 / (crop_bbox[2] - crop_bbox[0])
            proj_2D_loc = np.round(proj_2D_loc * ratio).astype(int)
            proj_2D_loc[proj_2D_loc >= 224] = 223
            proj_2D_loc[proj_2D_loc < 0] = 0
            # the visible points should be within mask
            valid_mask = mask[proj_2D_loc[:, 0], proj_2D_loc[:, 1]] == 1
            proj_2D_loc = proj_2D_loc[valid_mask]
            vis_pt_idx = vis_pt_idx[valid_mask]
            visible_pt_dict_processed[image_name]["vis_pt_idx"] = vis_pt_idx
            visible_pt_dict_processed[image_name]["proj_2D_loc"] = proj_2D_loc

            for part in os.listdir(old_label_dir):
                label = Image.open(os.path.join(old_label_dir, part, image_name)).convert('L')
                label = ImageOps.expand(label, border=padding, fill=0)
                label = label.crop(crop_bbox)
                label = label.resize((224, 224), Image.NEAREST)
                label_array = np.array(label) * mask
                label = Image.fromarray(label_array)
                label.save(os.path.join(new_label_dir, part, image_name))
        
        np.savez(os.path.join(out_dir, "visible_pt_dict_processed.npz"), **visible_pt_dict_processed)

        print(f"Running time: {time.time() - start_time} seconds")

    time.sleep(5)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help="<#TODO: path to the root rendered data directory (for labeling)>")
    parser.add_argument('--out_dir', type=str, help="<#TODO: path to the output directory (for training), e.g., fully_processed_data>")
    args = parser.parse_args()

    categories = [name for name in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, name))]

    num_cpus = min(os.cpu_count(), 6)
    print(f"Using {num_cpus} CPUs.")

    random.shuffle(categories)
    categories = split_list(categories, num_cpus)

    with ProcessPoolExecutor(max_workers=num_cpus) as main_executor:
        main_futures = [main_executor.submit(run_main, args, categories[i], i) for i in range(num_cpus)]