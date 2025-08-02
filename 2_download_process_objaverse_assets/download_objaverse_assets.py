import objaverse
import argparse
import json
import os
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--matches_dir', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

matches = json.load(open(args.matches_dir))
all_assets = []
for category, uids in matches.items():
    all_assets += uids
print(len(all_assets))

objaverse.BASE_PATH = args.save_dir
objaverse._VERSIONED_PATH = args.save_dir

objaverse_annotations = objaverse.load_annotations(all_assets)
objects = objaverse.load_objects(
    uids=all_assets,
    download_processes=multiprocessing.cpu_count()
)
