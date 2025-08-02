import os
import time
import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_square_bbox(bbox, pad_ratio = 1.5):
    # Unpack the original bounding box
    x0, y0, x1, y1 = bbox
    
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
    new_x0 = center_x - new_side / 2
    new_y0 = center_y - new_side / 2
    new_x1 = center_x + new_side / 2
    new_y1 = center_y + new_side / 2

    new_x0 = max(int(new_x0), 0)
    new_y0 = max(int(new_y0), 0)
    new_x1 = min(int(new_x1), 490)
    new_y1 = min(int(new_y1), 490)
    
    return [new_x0, new_y0, new_x1, new_y1]

def main(category, obj_dir, crop_requirement):
    category = category.split("---")[-1].replace(".glb","")
    if os.path.exists(os.path.join(obj_dir, "part_annotations")):
        bbox_dir = os.path.join(obj_dir, "part_annotations")
    else:
        # some categories may not be processed / used.
        return
    affordance_annos = [name for name in os.listdir(bbox_dir) if "aggregated" in name]
    # no affordance found
    if len(affordance_annos) == 0:
        return
    
    for i, affordance_anno in enumerate(affordance_annos):
        affordance = affordance_anno.removesuffix('_aggregated.json')

        # If we don't need to crop, skip it!
        if not crop_requirement[category][affordance.replace('_', ' ')]:
            continue

        dict_out_dir = os.path.join(obj_dir, "crop_offset")
        os.makedirs(dict_out_dir, exist_ok=True)
        
        out_dir = os.path.join(obj_dir, "rgb_images_cropped", affordance)
        os.makedirs(out_dir, exist_ok=True)
        if len(os.listdir(out_dir)) == 19: # HACK: already processed.
            continue
        
        bboxes = json.load(open(os.path.join(obj_dir, "part_annotations", affordance_anno)))
        images = os.listdir(os.path.join(obj_dir, "rgb_images"))

        offsets = {}
        for img in images:
            if img in bboxes:
                bbox = list(bboxes[img].values())[0]
                square_bbox = create_square_bbox(bbox)
                orig_image = Image.open(os.path.join(obj_dir, "rgb_images", img))
                cropped_image = orig_image.crop(square_bbox)
                cropped_image.save(os.path.join(out_dir, img))
                offsets[img] = [square_bbox[0], square_bbox[1]]

        offsets = json.dumps(offsets, indent=True)
        with open(os.path.join(dict_out_dir, f"{affordance}_crop_offset.json"), "w") as f:
            f.writelines(offsets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('category')
    parser.add_argument('--root_dir', type=str, help="path to the root rendered data directory")
    parser.add_argument('--crop_requirement_dir', type=str, help="path to crop_requirement.json")
    parser.add_argument('--all_category', action='store_true', help='run on all categories')
    args = parser.parse_args()

    crop_requirement = json.load(open(args.crop_requirement_dir))

    if args.all_category:
        categories = [name for name in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, name))]
    else:
        categories = args.category.split(",")

    start_time = time.time()
    for cat_idx, category in enumerate(categories):
        print(f"\n{cat_idx+1}/{len(categories)}: Running crop-padding on {category} images!")

        obj_dir = os.path.join(args.root_dir, category)
        main(category, obj_dir, crop_requirement)

        print(f"Running time: {time.time() - start_time} seconds")