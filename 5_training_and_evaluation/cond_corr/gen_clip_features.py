import os
import glob
import torch
import clip
import numpy as np
import copy
import argparse
import glob
import json
import argparse

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

@torch.no_grad
def main(args):
    taxonomy = json.load(open(args.taxonomy_file))

    affordances = []

    for k, v in taxonomy.items():
        affordances.extend(v)

    affordances = list(set(affordances))
    print(f"There are {len(affordances)} functions in total.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print([x.replace("_", " ").replace("-", " ") for x in affordances])
    affordance_tokens = clip.tokenize([x.replace("_", " ").replace("-", " ") for x in affordances]).to(device)
    affordance_feats = model.encode_text(affordance_tokens)
    
    affordance_feats /= affordance_feats.norm(dim=-1, keepdim=True)
    affordance_feats = affordance_feats.cpu()

    out_dir = os.path.join(args.data_root_path, "clip_text_features")
    os.makedirs(out_dir, exist_ok=True)

    for affordance, feat in zip(affordances, affordance_feats):
        torch.save(feat, os.path.join(out_dir, f"{affordance}.pth"))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxonomy_file', type=str, help="path to objaverse_object2actions.json")
    parser.add_argument('--data_root_path', type=str, help="path to dataset root")
    args = parser.parse_args()
    main(args)