import os
import random
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import pairwise_distances

import torch
import torchvision
import torch.nn as nn
import wandb
from tqdm import tqdm
import cv2

from torch.utils.data import DataLoader
from prettytable import PrettyTable
from PIL import Image
from datetime import datetime
from collections import defaultdict
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor

from cond_corr.model.correspondence_conditioner import MLPConditioner, DINO_embedder, ConditionedDino, NCESoftmaxLoss
from cond_corr.model.sd_dino_model import SD_DINO
from cond_corr.utils.viz_utils import make_viz, normalize_tensor, convert_tensor_image
from cond_corr.utils.evaluation import cal_sim
from cond_corr.train import visualize

def dist_nearest_neighbors(A, B):
    # Compute pairwise distances between A and B
    distances = pairwise_distances(A, B)
    
    # Find the minimum distance from each point in A to its nearest neighbor in B
    min_distances = np.min(distances, axis=1)
    
    return min_distances

def euclidean_distance(A, B):
    if len(A.shape) == 2:
        distances = np.sqrt(np.sum((A - B)**2, axis=1))
    elif len(A.shape) == 1:
        distances = np.sqrt(np.sum((A - B)**2))
    return distances

def normalize(array):
    normalized_array = (array - array.min()) / (array.max() - array.min())
    return normalized_array

def load_aff_text_features(clip_dir):
    feat_path = clip_dir
    items = os.listdir(feat_path)
    feat_dict = {}
    
    for item in items:
        item_path = os.path.join(feat_path, item)
        feat_dict[item.replace('.pth','')] = torch.load(item_path).cuda()

    return feat_dict

@torch.no_grad()
def find_cycle_consistent_matches(model, img1, mask1, img2, mask2, lang_feat, dino=False, sd=False, chance=False):
    if chance:
        row_indices, col_indices = np.where(mask1 == 1)
        co1_matches = np.array(list(zip(col_indices, row_indices)))
        row_indices, col_indices = np.where(mask2 == 1)
        indices2 = np.array(list(zip(col_indices, row_indices)))
        sample_indices = np.random.choice(len(indices2), size=len(co1_matches), replace=True)
        co2_matches = indices2[sample_indices]
        permuted_indices = np.random.permutation(len(co1_matches))
        return co1_matches[permuted_indices], co2_matches[permuted_indices]

    row_indices, col_indices = np.where(mask1 == 1)
    indices1 = np.array(list(zip(col_indices, row_indices)))
    f1_co = torch.tensor(indices1).cuda().unsqueeze(0)
    lang_feat = lang_feat.unsqueeze(0)
    if not sd:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    if dino and not sd:
        pred_maps = model.get_DINO_heatmap(img1, img2, f1_co)
    elif sd:
        pred_maps = model.get_heatmap(img1, img2, f1_co)
    else:
        pred_maps = model.get_heatmap(img1, img2, f1_co, lang_feat, lang_feat)

    pred_maps = pred_maps.cpu().numpy()
    pred_maps *= mask2
    pred_maps -= 100 * (1-mask2) # everywhere not on the mask is set to a very small value

    B, n, m = pred_maps.shape
    flat_indices = np.argmax(pred_maps.reshape(B, -1), axis=1)
    rows, cols = np.unravel_index(flat_indices, (n, m))
    max_coords = np.stack((cols, rows), axis=1)
    max_sims = pred_maps[np.arange(len(rows)),rows, cols]
    max_sims = normalize(max_sims)
    
    f2_co = torch.tensor(max_coords).cuda().unsqueeze(0)
    if dino and not sd:
        pred_maps_f1 = model.get_DINO_heatmap(img2, img1, f2_co)
    elif sd:
        pred_maps_f1 = model.get_heatmap(img2, img1, f2_co)
    else:
        pred_maps_f1 = model.get_heatmap(img2, img1, f2_co, lang_feat, lang_feat)

    pred_maps_f1 = pred_maps_f1.cpu().numpy()
    pred_maps_f1 *= mask1
    pred_maps_f1 -= 100 * (1-mask1) # everywhere not on the mask is set to a very small value

    B, n, m = pred_maps_f1.shape
    flat_indices = np.argmax(pred_maps_f1.reshape(B, -1), axis=1)
    rows, cols = np.unravel_index(flat_indices, (n, m))
    max_coords_f1 = np.stack((cols, rows), axis=1)

    distances = euclidean_distance(max_coords_f1, indices1)
    distances = normalize(distances)
    scores = - (1.0 - distances) * max_sims
    scores_indices = np.argsort(scores)

    return indices1[scores_indices, :], max_coords[scores_indices, :]

def calculate_precision(pred1, pred2, gt1, gt2, pck_threshold):
    gt_tree1 = cKDTree(gt1)
    gt_tree2 = cKDTree(gt2)

    # Check each predicted coordinate pair
    correct_matches = []
    for co1_fake, co2_fake in zip(pred1, pred2):
        # Get all indices within pck_threshold radius for each prediction
        idxs1 = gt_tree1.query_ball_point(co1_fake, pck_threshold)
        idxs2 = gt_tree2.query_ball_point(co2_fake, pck_threshold)
        
        # Find common indices in both idxs1 and idxs2
        common_idxs = set(idxs1).intersection(set(idxs2))
        
        # If there are common indices, we consider the prediction as correct
        if common_idxs:
            correct_matches.append(1)
        else:
            correct_matches.append(0)

    return np.array(correct_matches)

def calculate_recall(pred1, pred2, gt1, gt2, pck_threshold):
    pred_tree1 = cKDTree(pred1)
    pred_tree2 = cKDTree(pred2)
    recovered_matches = []

    for co1_real, co2_real in zip(gt1, gt2):
        idxs1 = pred_tree1.query_ball_point(co1_real, pck_threshold)
        idxs2 = pred_tree2.query_ball_point(co2_real, pck_threshold)
        common_idxs = set(idxs1).intersection(set(idxs2))
        if common_idxs:
            recovered_matches.append(min(common_idxs))
        else:
            recovered_matches.append(float("inf"))
    return np.array(recovered_matches)

def calculate_f1_score(precision, recall):
    # Check for zero division case
    if precision + recall == 0:
        return 0.0
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


@torch.no_grad()
def run_inference(model, lang_feat_dict, action, BASE_DIR, out_dir, anno_dir, metrics, affordance_dir=None, dino=False, sd=False, chance=False, viz=False):
    objs = os.listdir(anno_dir)
    obj1 = objs[0]
    obj2 = objs[1]
    obj1_name, obj1_img = obj1.replace(".npy", "").split("|||")
    obj2_name, obj2_img = obj2.replace(".npy", "").split("|||")
    obj1_points = np.load(os.path.join(anno_dir, obj1))
    obj2_points = np.load(os.path.join(anno_dir, obj2))

    img1 = os.path.join(BASE_DIR, obj1_name, "rgb_images_processed", obj1_img)
    mask1 = os.path.join(BASE_DIR, obj1_name, "object_masks_processed", obj1_img)
    img2 = os.path.join(BASE_DIR, obj2_name, "rgb_images_processed", obj2_img)
    mask2 = os.path.join(BASE_DIR, obj2_name, "object_masks_processed", obj2_img)

    img1 = Image.open(img1).convert("RGB")
    mask1 = Image.open(mask1).convert("L")
    img2 = Image.open(img2).convert("RGB")
    mask2 = Image.open(mask2).convert("L")

    image_size = 224
    mask1 = np.array(mask1)//255
    mask2 = np.array(mask2)//255
    # if not dino and not sd:
    if not sd:
        img1 = torchvision.transforms.functional.to_tensor(img1).cuda()
        img2 = torchvision.transforms.functional.to_tensor(img2).cuda()

    lang_feat = lang_feat_dict[action]
    co1 = obj1_points[:,[1,0]] # row, col --> col, row
    co2 = obj2_points[:,[1,0]] # row, col --> col, row
    
    if affordance_dir is not None:
        affordance1 = os.path.join(affordance_dir, obj1_name, "affordance_masks", action, obj1_img)
        affordance2 = os.path.join(affordance_dir, obj2_name, "affordance_masks", action, obj2_img)
        
        try:
            affordance1 = Image.open(affordance1).convert("L")
            affordance2 = Image.open(affordance2).convert("L")
        except:
            print("WARNING: no affordance mask found!!")
            affordance_dir = None
    
    if not sd and model is not None:
        if model.mask_loss:
            affordance_dir = "pred_mask"
            pred_mask1 = model.get_mask(img1.unsqueeze(0), lang_feat.unsqueeze(0))
            pred_mask1 = pred_mask1.cpu().numpy()
            pred_mask1 *= mask1
            affordance1 = pred_mask1 * 255
            pred_mask2 = model.get_mask(img2.unsqueeze(0), lang_feat.unsqueeze(0))
            pred_mask2 = pred_mask2.cpu().numpy()
            pred_mask2 *= mask2
            affordance2 = pred_mask2 * 255

    if affordance_dir is not None:
        affordance1 = np.array(affordance1)//255
        affordance2 = np.array(affordance2)//255

        fg_mask1 = mask1 * affordance1
        bg_mask1 = mask1 * (1-affordance1)
        fg_mask2 = mask2 * affordance2
        bg_mask2 = mask2 * (1-affordance2)
        # if any of the mask is too small, run normally
        if len(np.where(fg_mask1 == 1)[0])<16 or len(np.where(bg_mask1 == 1)[0])<16 or len(np.where(fg_mask2 == 1)[0])<16 or len(np.where(bg_mask2 == 1)[0])<16:
            co1_matches, co2_matches = find_cycle_consistent_matches(model, img1, mask1, img2, mask2, lang_feat, dino=dino, sd=sd, chance=chance)
        else:
            co1_matches_fg, co2_matches_fg = find_cycle_consistent_matches(model, img1, fg_mask1, img2, fg_mask2, lang_feat, dino=dino, sd=sd, chance=chance)
            co1_matches_bg, co2_matches_bg = find_cycle_consistent_matches(model, img1, bg_mask1, img2, bg_mask2, lang_feat, dino=dino, sd=sd, chance=chance)
            co1_matches = np.vstack((co1_matches_fg, co1_matches_bg))
            co2_matches = np.vstack((co2_matches_fg, co2_matches_bg))
    else:
        co1_matches, co2_matches = find_cycle_consistent_matches(model, img1, mask1, img2, mask2, lang_feat, dino=dino, sd=sd, chance=chance)

    pck_thresholds = [23,15,10,5]
    for pck_threshold in pck_thresholds:
        correct_matches = calculate_precision(co1_matches, co2_matches, co1, co2, pck_threshold)
        recovered_matches = calculate_recall(co1_matches, co2_matches, co1, co2, pck_threshold)
        
        n_pts = len(co1_matches)
        precisions = []
        recalls = []
        f1s = []
        
        for cutoff in np.arange(0.01, 1.01, 0.01):
            top_k = int(n_pts * cutoff)
            precision = np.sum(correct_matches[:top_k]) / top_k
            recall = np.mean(recovered_matches < top_k)
            f1_score = calculate_f1_score(precision, recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1_score)
        
        metrics[pck_threshold]["precision"].append(precisions)
        metrics[pck_threshold]["recall"].append(recalls)
        metrics[pck_threshold]["f1"].append(f1s)

    # When visualizing, no need to run for the other direction.
    # if not sd:
    #     image1 = np.array(torchvision.transforms.functional.to_pil_image(img1))
    #     image2 = np.array(torchvision.transforms.functional.to_pil_image(img2))
    # else:
    #     image1 = np.array(img1)
    #     image2 = np.array(img2)
    # img = np.concatenate((image1, image2), axis=1)
    # plt.figure()
    # plt.imshow(img)
    # cmap = plt.get_cmap('jet')
    # H0, W0, H1, W1 = *image1.shape[:2], *image2.shape[:2]

    # co1, co2 = co1_matches, co2_matches
    # top_k_indices = [] # separated by 5 pixels
    # for index in range(len(co1)):
    #     seen = False
    #     for prev in top_k_indices:
    #         if seen:
    #             break
    #         if euclidean_distance(co1[index], co1[prev]) < 5.0:
    #             seen = True
    #     for prev in top_k_indices:
    #         if seen:
    #             break
    #         if euclidean_distance(co2[index], co2[prev]) < 5.0:
    #             seen = True
    #     if not seen:
    #         top_k_indices.append(index)
    #     if len(top_k_indices) >= 10:
    #         break
    
    # co1 = co1[top_k_indices]
    # co2 = co2[top_k_indices]
    # sorted_indices = np.argsort(co1[:, 1])
    # co1 = co1[sorted_indices]
    # co2 = co2[sorted_indices]

    # for j, ((x1, y1), (x2, y2)) in enumerate(zip(co1, co2)):
    #     if len(co1) <= 1: 
    #         c = cmap(0)
    #     else:
    #         c = cmap(j / (len(co1)-1))
    #     plt.plot([x1, x2 + W0], [y1, y2], '-+', color=c, scalex=False, scaley=False)
    # plt.tight_layout()
    # out_name = f"{action}|||{obj1_name}|||{obj1_img}|||{obj2_name}|||{obj2_img}|||trial_{viz}"
    # out_name += ".png"
    # plt.savefig(os.path.join(out_dir, out_name))
    # plt.close()

    # now, do it for the other direction
    if affordance_dir is not None:
        # if any of the mask is too small, run normally
        if len(np.where(fg_mask1 == 1)[0])<16 or len(np.where(bg_mask1 == 1)[0])<16 or len(np.where(fg_mask2 == 1)[0])<16 or len(np.where(bg_mask2 == 1)[0])<16:
            co2_matches, co1_matches = find_cycle_consistent_matches(model, img2, mask2, img1, mask1, lang_feat, dino=dino, sd=sd, chance=chance)
        else:
            co2_matches_fg, co1_matches_fg = find_cycle_consistent_matches(model, img2, fg_mask2, img1, fg_mask1, lang_feat, dino=dino, sd=sd, chance=chance)
            co2_matches_bg, co1_matches_bg = find_cycle_consistent_matches(model, img2, bg_mask2, img1, bg_mask1, lang_feat, dino=dino, sd=sd, chance=chance)
            co2_matches = np.vstack((co2_matches_fg, co2_matches_bg))
            co1_matches = np.vstack((co1_matches_fg, co1_matches_bg))
    else:
        co2_matches, co1_matches = find_cycle_consistent_matches(model, img2, mask2, img1, mask1, lang_feat, dino=dino, sd=sd, chance=chance)
    
    pck_thresholds = [23,15,10,5]
    for pck_threshold in pck_thresholds:
        correct_matches = calculate_precision(co2_matches, co1_matches, co2, co1, pck_threshold)
        recovered_matches = calculate_recall(co2_matches, co1_matches, co2, co1, pck_threshold)
        
        n_pts = len(co2_matches)
        precisions = []
        recalls = []
        f1s = []

        for cutoff in np.arange(0.01, 1.01, 0.01):
            top_k = int(n_pts * cutoff)
            precision = np.sum(correct_matches[:top_k]) / top_k
            recall = np.mean(recovered_matches < top_k)
            f1_score = calculate_f1_score(precision, recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1_score)

        metrics[pck_threshold]["precision"].append(precisions)
        metrics[pck_threshold]["recall"].append(recalls)
        metrics[pck_threshold]["f1"].append(f1s)

    return metrics
    
def main(args):
    BASE_DIR = "<#TODO: path to your base dir for dataset>/fully_processed_data>"
    EXP_LOGS_DIR = "<#TODO: path to your experiment_logs directory>"
    
    ckpt_dirs = []
    if args.ckpt_dir == "all":
        experiment_dir = EXP_LOGS_DIR
        experiments = os.listdir(experiment_dir)
        experiments = [x for x in experiments if "Dino" in x]
        for experiment in experiments:
            timecode = os.listdir(os.path.join(experiment_dir, experiment))[0]
            ckpt_dirs.append(os.path.join(experiment_dir, experiment, timecode, "checkpoints", "epoch_0100.pt"))
    else:
        ckpt_dirs.append(args.ckpt_dir)

    print(ckpt_dirs)
    
    for ckpt_dir in ckpt_dirs:
        print(f"\nProcessing {ckpt_dir}")
        start_time = time.time()

        annotations = json.load(open(args.annotation_list))
        # seeding
        SEED = 1234  
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # paths
        checkpoint_path = ckpt_dir
        epoch = checkpoint_path.split('/')[-1].replace(".pt", "")
        log_dir = '/'.join(checkpoint_path.split('/')[:-2])

        # setup model
        if args.chance:
            model = None
        elif args.sd:
            model = SD_DINO(sd=args.sd, dino=args.dino)
        elif args.dino:
            model = ConditionedDino(args.use_last_layers, args.use_smaller_stride, args.use_featup)
            model = model.cuda()
            model.eval()
        else:
            additional_model_args = {"use_film": args.film, "use_lora": args.lora, "lora_rank": 8, "n_blocks": 3}
            if args.dino_size is not None:
                additional_model_args["dino_size"] = args.dino_size
                
            model = ConditionedDino(args.use_last_layers, args.use_smaller_stride, args.use_featup, mask_loss=args.pred_mask, additional_model_args=additional_model_args)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
            model.load_state_dict(checkpoint)
            model = model.cuda()
            model.eval()
        
        lang_feat_dict = load_aff_text_features("<#TODO: path to CLIP text features>")

        out_dir = os.path.join(log_dir, "2D_eval_viz", f"objaverse_test_seen_{epoch}")
        if args.sd:
            out_dir = os.path.join(f"{EXP_LOGS_DIR}/SD", "2D_eval_viz", "objaverse_test_seen")
        if args.dino:
            out_dir = os.path.join(f"{EXP_LOGS_DIR}/Dino", "2D_eval_viz", "objaverse_test_seen")
        if args.sd and args.dino:
            out_dir = os.path.join(f"{EXP_LOGS_DIR}/SD-Dino", "2D_eval_viz", "objaverse_test_seen")
        if args.affordance_dir is not None:
            # HACK: could also be other combinations
            if "seen_test_images" in args.affordance_dir:
                if args.dino: 
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Dino-CogVLM", "2D_eval_viz", "objaverse_test_seen")
                else:
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Spatial_fullImg_bgAug-CogVLM", "2D_eval_viz", "objaverse_test_seen")
            elif "seen_test_manipvqa_part" in args.affordance_dir:
                if args.dino: 
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Dino-ManipVQA-Part", "2D_eval_viz", "objaverse_test_seen")
                else:
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Spatial_fullImg_bgAug-ManipVQA-Part", "2D_eval_viz", "objaverse_test_seen")
            elif "seen_test_manipvqa_action" in args.affordance_dir:
                if args.dino: 
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Dino-ManipVQA-Action", "2D_eval_viz", "objaverse_test_seen")
                else:
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Spatial_fullImg_bgAug-ManipVQA-Action", "2D_eval_viz", "objaverse_test_seen")
        if args.chance:
            if args.affordance_dir is not None:
                if "seen_test_images" in args.affordance_dir:
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Chance-CogVLM", "2D_eval_viz", "objaverse_test_seen")
                elif "seen_test_manipvqa_part" in args.affordance_dir:
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Chance-ManipVQA-Part", "2D_eval_viz", "objaverse_test_seen")
                elif "seen_test_manipvqa_action" in args.affordance_dir:
                    out_dir = os.path.join(f"{EXP_LOGS_DIR}/Chance-ManipVQA-Action", "2D_eval_viz", "objaverse_test_seen")
            else:
                out_dir = os.path.join(f"{EXP_LOGS_DIR}/Chance", "2D_eval_viz", "objaverse_test_seen")
        print(f"\nOut directory: {out_dir}\n")

        os.makedirs(out_dir, exist_ok=True)
        metrics = {}
        pck_thresholds = [23,15,10,5]
        for pck_threshold in pck_thresholds:
            metrics[pck_threshold] = {"precision": [], "recall": [], "f1": []}

        for action, entries in annotations.items():
            for obj_pair, trials in entries.items():
                for idx, trial in enumerate(trials):
                    anno_dir = os.path.join(args.annotation_dir, action, obj_pair+"|||0000", f"trial_{trial}")
                    metrics = run_inference(model, lang_feat_dict, action, BASE_DIR, out_dir, anno_dir, metrics, affordance_dir=args.affordance_dir, dino=args.dino, sd=args.sd, chance=args.chance, viz=trial)

        out_str = json.dumps(metrics, indent=True)
        with open(os.path.join(out_dir, "discovery_metrics.json"), "w") as f:
            f.writelines(out_str)

        for pck_threshold in pck_thresholds:
            f1_scores = metrics[pck_threshold]["f1"]
            best_f1_score = []
            for f1_score in f1_scores:
                best_f1_score.append(np.max(f1_score))
            print(f"The average best f1 score @ {pck_threshold} pixels is {np.mean(best_f1_score):.3f}.")

        print(f"Took {time.time() - start_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, help="path to the model ckpt")
    parser.add_argument('--annotation_list', type=str, help="path to selected_annotations_seen.json")
    parser.add_argument('--annotation_dir', type=str, help="path to annotations")
    parser.add_argument('--affordance_dir', type=str, help="for methods that predict functional part mask, put directory of predicted masks here")
    parser.add_argument('--use_last_layers', type=int, default=3)
    parser.add_argument('--use_smaller_stride', action='store_true')
    parser.add_argument('--use_featup', action='store_true')
    parser.add_argument('--dino', action='store_true', help="Run eval for DINO")
    parser.add_argument('--sd', action='store_true', help="Run eval for SD")
    parser.add_argument('--chance', action='store_true', help="Run eval for chance baseline")
    parser.add_argument('--pred_mask', action='store_true', help="whether our model predicts a mask or not")
    parser.add_argument('--film', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--dino_size', type=str, help="default should be base")
    args = parser.parse_args()
    main(args)