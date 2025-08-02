import os
import random
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import auc
import pandas as pd

BASE_DIR = "<#TODO: path to your experiment_logs directory>"
experiments = os.listdir(BASE_DIR)
experiments = [x for x in experiments if "Aug" in x]
experiments.sort()
# TODO: adjust this list for the runs you have
baselines = ["Chance", "Chance-CogVLM", "Dino", "SD", "SD-Dino", "Dino-CogVLM", "Chance-ManipVQA-Part", "Chance-ManipVQA-Action", "Dino-ManipVQA-Part", "Dino-ManipVQA-Action"]
experiments = baselines + experiments
annotation_dir = "<#TODO: path to your HANDAL selected_annotations_seen.json directory>"
annotations = json.load(open(annotation_dir))

keys = ["name", "epoch", "norm_dist", "norm_dist_within", "norm_dist_across"]
keys += ["PCK_23", "PCK_23_within", "PCK_23_across", "PCK_15", "PCK_15_within", "PCK_15_across", "PCK_10", "PCK_10_within", "PCK_10_across"]
keys += ["mean_best_f1_23", "mean_best_f1_23_within", "mean_best_f1_23_across", "mean_best_f1_15", "mean_best_f1_15_within", "mean_best_f1_15_across"]
keys += ["mean_best_f1_10", "mean_best_f1_10_within", "mean_best_f1_10_across", "mean_best_f1_5", "mean_best_f1_5_within", "mean_best_f1_5_across"]
keys += ["mAP_23", "mAP_23_within", "mAP_23_across", "mAP_15", "mAP_15_within", "mAP_15_across"]
keys += ["mAP_10", "mAP_10_within", "mAP_10_across", "mAP_5", "mAP_5_within", "mAP_5_across"]
aggregated_results = []

def average_precision_score(precision, recall):
    precision = np.insert(np.array(precision), 0, 1.0)
    recall = np.insert(np.array(recall), 0, 0.0)
    return np.sum((recall[1:] - recall[:-1]) * precision[1:])

for experiment in experiments:
    print(experiment)
    
    if experiment in baselines:
        result_dir = os.path.join(BASE_DIR, experiment)
    else:
        result_dir = os.path.join(BASE_DIR, experiment)
        timecode = sorted(os.listdir(result_dir))[-1]  # always take last one
        result_dir = os.path.join(result_dir, timecode)
    
    if os.path.exists(os.path.join(result_dir, "2D_eval_viz")):
        epochs = os.listdir(os.path.join(result_dir, "2D_eval_viz"))
    elif os.path.exists(os.path.join(result_dir, "2D_discovery_viz")):
        epochs = os.listdir(os.path.join(result_dir, "2D_discovery_viz"))
    else:
        epochs = []
        
    epochs = [x for x in epochs if x == "handal_test_seen" or "handal_test_seen_epoch_" in x]
    epochs.sort()

    for epoch in epochs:
        exp_result = {key: np.nan for key in keys}
        exp_result["name"] = experiment
        if experiment not in baselines:
            if epoch == "handal_test":
                exp_result["epoch"] = 100 # HACK
            else:
                exp_result["epoch"] = int(epoch.split("_")[-1])

        # label transfer
        suffix = f"2D_eval_viz/{epoch}/label_transfer_metrics.json"
        if os.path.exists(os.path.join(result_dir, suffix)):
            metrics = json.load(open(os.path.join(result_dir, suffix)))
            metrics = [np.array(x) for x in metrics]

            results = {
                "within": {"norm_dist": [], "PCK_23": [], "PCK_15": [], "PCK_10": []},
                "across": {"norm_dist": [], "PCK_23": [], "PCK_15": [], "PCK_10": []}
            }

            # Function to update metrics for a given category
            def update_metrics(category, metric_data):
                results[category]["norm_dist"].extend([np.mean(metric_data[0]) / 223, np.mean(metric_data[1]) / 223])
                results[category]["PCK_23"].extend([np.mean(metric_data[0] <= 23), np.mean(metric_data[1] <= 23)])
                results[category]["PCK_15"].extend([np.mean(metric_data[0] <= 15), np.mean(metric_data[1] <= 15)])
                results[category]["PCK_10"].extend([np.mean(metric_data[0] <= 10), np.mean(metric_data[1] <= 10)])

            count = 0
            for action, entries in annotations.items():
                for obj_pair, trials in entries.items():
                    obj1, obj2 = obj_pair.split("|||")
                    obj1_category = obj1.split("---")[0]
                    obj2_category = obj2.split("---")[0]

                    # Determine whether the metrics should be classified as "within" or "across"
                    category = "within" if obj1_category == obj2_category else "across"

                    # Update metrics for the given category
                    for _ in range(len(trials)):
                        update_metrics(category, [metrics[count], metrics[count + 1]])
                        count += 2

            assert(count == len(metrics))

            for metric_name in ["norm_dist", "PCK_23", "PCK_15", "PCK_10"]:
                exp_result[f"{metric_name}_within"] = np.mean(results["within"][metric_name])
                exp_result[f"{metric_name}_across"] = np.mean(results["across"][metric_name])
                exp_result[f"{metric_name}"] = np.mean(results["within"][metric_name] + results["across"][metric_name])

        # discovery
        suffix = f"2D_discovery_viz/{epoch}/discovery_metrics.json"
        if os.path.exists(os.path.join(result_dir, suffix)):
            discovery_metric = json.load(open(os.path.join(result_dir, suffix)))
            for pck_threshold, metrics in discovery_metric.items():
                results = {
                    "within": {"mean_best_f1": [], "mAP": []},
                    "across": {"mean_best_f1": [], "mAP": []}
                }
                count = 0
                for action, entries in annotations.items():
                    for obj_pair, trials in entries.items():
                        obj1, obj2 = obj_pair.split("|||")
                        obj1_category = obj1.split("---")[0]
                        obj2_category = obj2.split("---")[0]

                        # Determine whether the metrics should be classified as "within" or "across"
                        category = "within" if obj1_category == obj2_category else "across"

                        # Update metrics for the given category
                        for _ in range(len(trials)):
                            results[category]["mean_best_f1"].append(np.max(metrics["f1"][count]))
                            results[category]["mean_best_f1"].append(np.max(metrics["f1"][count+1]))
                            results[category]["mAP"].append(average_precision_score(metrics["precision"][count], metrics["recall"][count]))
                            results[category]["mAP"].append(average_precision_score(metrics["precision"][count+1], metrics["recall"][count+1]))
                            count += 2
                for metric_name in ["mean_best_f1", "mAP"]:
                    exp_result[f"{metric_name}_{pck_threshold}_within"] = np.mean(results["within"][metric_name])
                    exp_result[f"{metric_name}_{pck_threshold}_across"] = np.mean(results["across"][metric_name])
                    exp_result[f"{metric_name}_{pck_threshold}"] = np.mean(results["within"][metric_name] + results["across"][metric_name])
        
        aggregated_results.append(exp_result)

df = pd.DataFrame(aggregated_results)
df.to_csv('./handal_aggregated_results.csv', index=False)
                        
