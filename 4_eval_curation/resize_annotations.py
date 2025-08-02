import os
import time
import json
import argparse
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

mode = "seen"
base_dir = "<#TODO: path to the base diirectory for data>"
IMAGE_PATH = f"{base_dir}/<#TODO: path to the rendered training images before resizing>"
orig_dir = f"{base_dir}/annotations/2d_annotations/{mode}"
out_dir = f"{base_dir}/annotations/2d_annotations_resized/{mode}"

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

for action in os.listdir(orig_dir):
    for obj_pair in os.listdir(os.path.join(orig_dir, action)):
        for trial in os.listdir(os.path.join(orig_dir, action, obj_pair)):
            objs = os.listdir(f"{orig_dir}/{action}/{obj_pair}/{trial}")
            obj1 = objs[0]
            obj2 = objs[1]
            obj1_name, obj1_img = obj1.replace(".npy", "").split("|||")
            obj2_name, obj2_img = obj2.replace(".npy", "").split("|||")
            obj1_points = np.load(os.path.join(f"{orig_dir}/{action}/{obj_pair}/{trial}", obj1))
            obj2_points = np.load(os.path.join(f"{orig_dir}/{action}/{obj_pair}/{trial}", obj2))
            
            mask1 = Image.open(os.path.join(IMAGE_PATH, obj1_name, "segmentation", obj1_img.replace(".png", "_0001.png"))).convert('L')
            mask2 = Image.open(os.path.join(IMAGE_PATH, obj2_name, "segmentation", obj2_img.replace(".png", "_0001.png"))).convert('L')
            
            # pad it to larger image
            padding = (
                60,  # left padding
                60   # top padding
            )
            mask1 = ImageOps.expand(mask1, border=padding, fill=0)
            mask2 = ImageOps.expand(mask2, border=padding, fill=0)
            crop_bbox1 = create_square_bbox(mask1)
            crop_bbox2 = create_square_bbox(mask2)

            # process point coordinates
            obj1_points[:, 0] += 60
            obj1_points[:, 1] += 60
            obj1_points[:, 0] -= crop_bbox1[1]
            obj1_points[:, 1] -= crop_bbox1[0]
            ratio1 = 224 / (crop_bbox1[2] - crop_bbox1[0])
            obj1_points = np.round(obj1_points * ratio1).astype(int)
            obj1_points[obj1_points >= 224] = 223
            obj1_points[obj1_points < 0] = 0

            obj2_points[:, 0] += 60
            obj2_points[:, 1] += 60
            obj2_points[:, 0] -= crop_bbox2[1]
            obj2_points[:, 1] -= crop_bbox2[0]
            ratio2 = 224 / (crop_bbox2[2] - crop_bbox2[0])
            obj2_points = np.round(obj2_points * ratio2).astype(int)
            obj2_points[obj2_points >= 224] = 223
            obj2_points[obj2_points < 0] = 0

            # save
            os.makedirs(os.path.join(out_dir, action, obj_pair, trial), exist_ok=True)
            np.save(os.path.join(out_dir, action, obj_pair, trial, obj1), obj1_points)
            np.save(os.path.join(out_dir, action, obj_pair, trial, obj2), obj2_points)
