import os
import numpy as np
import json
import glob
import cv2

colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Lime
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (192, 192, 192),  # Silver
    (128, 0, 0),  # Maroon
    (128, 128, 0),  # Olive
    (0, 128, 0),  # Green
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (0, 0, 128),  # Navy
    (255, 165, 0),  # Orange
    (255, 215, 0),  # Gold
    (75, 0, 130),  # Indigo
    (255, 20, 147),  # Deep Pink
]

def get_aff_json_name(affordance, agg_type):
    if agg_type == "aggregated":
        return f"{affordance}_aggregated.json"
    elif agg_type == "none":
        return f"{affordance}.json"
    
def load_mask(path):
    mask = cv2.imread(path)
    mask = mask[:,:,0]
    mask = mask//255
    return mask

def draw_bounding_box(image, color, bbox):
    """
    Draws a bounding box on an image.

    :param image: OpenCV image where the bounding box will be drawn
    :param color: Tuple of BGR color (blue, green, red)
    :param bbox: List or tuple of bounding box coordinates [x0, y0, x1, y1]
    """
    # OpenCV uses BGR, so convert the RGB color to BGR
    bgr_color = (color[2], color[1], color[0])

    # Draw the rectangle on the image
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bgr_color, thickness=2)

    return image

def check_valid_bbox_anno(mask, bbox, min_pixels=128, overlap_thresh=0.1):
    xmin, ymin, xmax, ymax = bbox
    roi = mask[ymin:ymax, xmin:xmax]
    box_area = (xmax-xmin) * (ymax-ymin)
    if box_area > 0:
        overlap = roi.sum() / box_area
    else:
        return False
    if (roi.sum() > min_pixels) and (overlap > overlap_thresh):
        return True
    else:
        return False
