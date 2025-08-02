import os
import time
import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
from concurrent.futures import ProcessPoolExecutor


def most_common_bbox(bboxes): # n_clusters=3
    """
    Identifies the most commonly occurring bounding box in a list where similar boxes are considered.

    Parameters:
    - bboxes: List of tuples/lists [x0, y0, x1, y1] representing bounding boxes.
    - n_clusters: Number of clusters to use in k-means clustering (default is 3).

    Returns:
    - The average bounding box of the most common cluster as [x0, y0, x1, y1].
    """
    if not bboxes:
        return None

    # Calculate centers of bounding boxes
    centers = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in bboxes])

    # Clustering, try n_clusters=2,3,4
    sil_score_max = -1
    best_n_clusters = 1
    for n_clusters in range(2, min(len(bboxes),5)):
        model = KMeans(n_clusters = n_clusters)
        labels = model.fit_predict(centers)
        try:
            sil_score = silhouette_score(centers, labels)
        except Exception as e:
            print(e)
            sil_score = -1
        # print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters

    kmeans = KMeans(n_clusters=best_n_clusters)
    labels = kmeans.fit_predict(centers)

    # Find the largest cluster
    largest_cluster_index = np.argmax(np.bincount(labels))

    # Filter bounding boxes in the largest cluster
    cluster_bboxes = np.array(bboxes)[labels == largest_cluster_index]

    # Calculate the average bounding box for this cluster
    avg_bbox = np.mean(cluster_bboxes, axis=0)
    
    avg_bbox = list(avg_bbox.astype(int))
    avg_bbox = [int(b) for b in avg_bbox]
    return avg_bbox

def aggregate_bbox(query_to_bbox):
    for key, value in query_to_bbox.items():
        new_value = most_common_bbox(value)
        if new_value is not None:
            query_to_bbox[key] = new_value
        else:
            query_to_bbox[key] = []
    return query_to_bbox

def main(category, part_dir):
    img_dir = os.path.join(part_dir, "rgb_images")
    mask_dir = os.path.join(part_dir, "object_masks")
    mode = "part_annotations"
    # mode = "part_annotations_cropped"
    if os.path.exists(os.path.join(part_dir, mode)):
        bbox_dir = os.path.join(part_dir, mode)
    else:
        # some categories may not be processed / used.
        return
    affordance_annos = [name for name in os.listdir(bbox_dir) if "aggregated" not in name]
    # no affordance found
    if len(affordance_annos) == 0:
        return
    
    for i, affordance_anno in enumerate(affordance_annos):
        affordance = affordance_anno.removesuffix('.json')
        # already processed
        if os.path.exists(os.path.join(bbox_dir, f"{affordance}_aggregated.json")):
            continue
        part_viz_dir = os.path.join(part_dir, "part_viz", affordance)
        os.makedirs(part_viz_dir, exist_ok=True)
        with open(os.path.join(bbox_dir, affordance_anno)) as f:
            img_to_bbox = json.load(f)
        
        new_img_to_bbox = {}
        for img_name, info in img_to_bbox.items():
            # update annotation
            new_img_to_bbox[img_name] = aggregate_bbox(info) 
            aggregated_bbox = list(new_img_to_bbox[img_name].values())
            
            # combine multiple queries and mask
            img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
            if os.path.exists(mask_dir):
                obj_mask = np.array(Image.open(os.path.join(mask_dir, img_name)).convert("L"))
                obj_mask = obj_mask / 255
            else:
                obj_mask = np.ones(img.size) # if no mask, all 1
            new_mask = np.zeros_like(obj_mask).astype(float)
            for b in aggregated_bbox:
                if b != []:
                    new_mask[b[1]:b[3]+1, b[0]:b[2]+1] += 1/len(aggregated_bbox)
            new_mask *= obj_mask
            new_mask = Image.fromarray((new_mask * 255).astype(np.uint8), mode='L')

            # visualize the result
            transparent_img = Image.blend(img, Image.new("RGB", img.size, (0,0,0)), alpha=0.9)
            img_masked = Image.composite(img, transparent_img, new_mask)
            img_masked.save(os.path.join(part_viz_dir, img_name))
            
        anno = json.dumps(new_img_to_bbox, indent=True)
        with open(os.path.join(bbox_dir, f"{affordance}_aggregated.json"), "w") as f:
            f.writelines(anno)
                
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

def run_main(args, categories, cluster_id):
    time.sleep(cluster_id*1) # hopefully avoid different processes writing at the same time...
    start_time = time.time()
    for cat_idx, category in enumerate(categories):
        print(f"\n{cat_idx+1}/{len(categories)}: Running aggregating part bbox on {category} images!")

        part_dir = os.path.join(args.root_dir, category)
        main(category, part_dir)

        print(f"Running time on {category}: {time.time() - start_time} seconds")
    time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help="path to the root rendered data directory")
    args = parser.parse_args()

    categories = [name for name in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, name))]
    num_cpus = min(os.cpu_count(), 8)
    print(f"Using {num_cpus} CPUs.")
    categories = split_list(categories, num_cpus)

    with ProcessPoolExecutor(max_workers=num_cpus) as main_executor:
        main_futures = [main_executor.submit(run_main, args, categories[i], i) for i in range(num_cpus)]