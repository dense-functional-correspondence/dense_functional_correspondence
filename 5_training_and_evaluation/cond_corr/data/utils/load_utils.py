import io
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import time
from pympler import asizeof


def load_file_to_ram(fpath):
    with open(fpath, 'rb') as file:
        image_bytes = file.read()
    # Create an in-memory byte stream from the read bytes
    byte_stream = io.BytesIO(image_bytes)
    return byte_stream

def load_data_in_ram(data_list):
    def load_item(item):
        item['img_path'] = load_file_to_ram(item['img_path'])
        item['mask_path'] = load_file_to_ram(item['mask_path'])
        if "bboxes" in item and isinstance(item["bboxes"][0], str):
            if os.path.exists(item["bboxes"][0]):
                item['bboxes'] = [load_file_to_ram(bbox) for bbox in item['bboxes']]

        return item
    
    print("Loading data in RAM...")
    start_time = time.time()
    ram_data_list = Parallel(n_jobs=16)(delayed(load_item)(item) for item in data_list)
    end_time = time.time()
    try:
        size = asizeof.asizeof(ram_data_list)
        size = size/(1024**3)
    except:
        size = 0.0
    print(f"Loaded {len(ram_data_list)} items of {size:.2f} GB in {end_time-start_time:.2f} seconds")

    return ram_data_list 

