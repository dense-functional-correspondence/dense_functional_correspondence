import torch
import torchvision
import numpy as np
import json
import random
import os
import glob
import time
from torchvision import transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from cond_corr.data.utils.aug_utils import HorizontalFlip, RandomCrop, resize_image_mask_box, pad_to_square
from cond_corr.data.utils.load_utils import load_data_in_ram
from cond_corr.data.base_loader import BaseCogVLM_loader, CogVLM_3D_loader

DATA_ROOT = "<#TODO: path to the dataset root>"
METADATA_PATH = "<#TODO: path to the dataset metadata>"  # e.g. "../0_taxonomy_and_metadata/Objaverse/"

class Objaverse_CC_cogvlm(CogVLM_3D_loader):
    def __init__(self, split="", image_size=224, load_in_ram=True, mode="train",
                num_points=128, test_selected=False, subsample=None, aug_params=None):
        super().__init__(
            split=split, 
            image_size=image_size, 
            load_in_ram=load_in_ram, 
            mode=mode,
            num_points=num_points,
            test_selected = test_selected,
            subsample = subsample,
            clip_dir = "<#TODO: path to CLIP text features>",
            aug_params = aug_params)

    @staticmethod
    def load_dataset(split, test_selected):
        start_time = time.time()

        if test_selected:
            assert(split == "test")
            data_list = os.path.join(METADATA_PATH, f"{split}_selected_list.json")
        else:
            data_list = os.path.join(METADATA_PATH, f"{split}_list.json")
        data_list = json.load(open(data_list))

        # data_list = data_list[:2000] # HACK for debugging only!!
        
        loaded_affs = set()
        data_dict_list = []

        print("")
        for item in data_list:
            obj, im_f, aff, bbox_idxs = item
            obj_category = obj.split("---")[-1].replace(".glb", "")

            item_name = f"{aff}||{obj}||{im_f}"
            img_path = os.path.join(DATA_ROOT, "fully_processed_data", obj, "rgb_images_processed", im_f)
            mask_path = os.path.join(DATA_ROOT, "fully_processed_data", obj, "object_masks_processed", im_f)
            assert os.path.exists(img_path)
            assert os.path.exists(mask_path)

            bboxes = [] # HACK: instead of actual bbox coordinates, we have the part mask file name
            for bbox_idx in bbox_idxs:
                bbox_dir = os.path.join(DATA_ROOT, "fully_processed_data", obj, "pseudo_labels_processed", bbox_idx, im_f)
                assert os.path.exists(bbox_dir)
                bboxes.append(bbox_dir)

            dct = {
                "name":item_name,
                "img_path":img_path,
                "mask_path":mask_path,
                "bboxes":bboxes,
                "bbox_idxs":bbox_idxs
                }

            data_dict_list.append(dct)
            print(f"Building data list: {len(data_dict_list):06d}", end='\r')
            loaded_affs.add(f"{aff}-{obj_category}")

        print(f"loaded {len(loaded_affs)} obj/aff combinations")
        print(f"Loading data set took {time.time()-start_time:.2f} seconds.")
        return data_dict_list
    

if __name__ == "__main__":
    
    SEED = 1234     
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    import matplotlib.pyplot as plt
    from cond_corr.data.sampler import BatchSampler

    split = "train"
    n_iter = 100
    n_pair = 2
    batch_size = 256

    aug_params = {
        "HorizontalFlip": [0.0],
        "VerticalFlip": [0.0],
        "BrightnessAug": [0.5, 0.6, 1.6],
        "HueAug": [0.5, -0.5, 0.5],
        "GammaAug": [0.5, 0.5, 1.5],
        "ContrastAug": [0.5, 0.6, 1.6],
        "SaturationAug": [0.5, 0.0, 2.0],
        "BlurAug": [0.0, [5, 7], 0.1, 3],
        "RandomMaskingAug": [1.0, 0.5, "domain_randomized", ["rotation", "scaling"]]
    }
    dataset = Objaverse_CC_cogvlm(split=split, load_in_ram=True, mode="train", test_selected=False, aug_params=aug_params)
    sampler = BatchSampler(dataset.affordance_label_dict, n_iter, n_pair, batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8)
    sampler.rng.seed(SEED)

    out_dir = f"<#TODO: your dir>/experiment_logs/data_samples/objaverse_loader_outputs_{split}"
    os.makedirs(out_dir, exist_ok=True)

    for batch in loader:
        n_items = len(batch['img'])
        for i in range(0,n_items - 1,2):
            print(i)

            img1 = batch['img'][i]
            img2 = batch['img'][i+1]

            hmap1 = batch['gt_hmap_mask'][i]
            hmap2 = batch['gt_hmap_mask'][i+1]

            pos_co1 = batch['pos_co'][i]
            pos_co2 = batch['pos_co'][i+1]

            neg_co1 = batch['neg_co'][i]
            neg_co2 = batch['neg_co'][i+1]


            name1 = batch['name'][i]
            name2 = batch['name'][i+1]

            print(name1, name2)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1,ncols=4)
            ax1.imshow(torchvision.transforms.functional.to_pil_image(img1))
            ax2.imshow(torchvision.transforms.functional.to_pil_image(img2))

            ax1.scatter(pos_co1[:,0], pos_co1[:,1], s=1, c='g')
            ax2.scatter(pos_co2[:,0], pos_co2[:,1], s=1, c='g')

            ax1.scatter(neg_co1[:,0], neg_co1[:,1], s=1, c='r')
            ax2.scatter(neg_co2[:,0], neg_co2[:,1], s=1, c='r')


            ax3.imshow(torchvision.transforms.functional.to_pil_image(hmap1))
            ax4.imshow(torchvision.transforms.functional.to_pil_image(hmap2))

            fig.suptitle(f"{name1}\n{name2}")
            fig.set_size_inches(10,4)
            fig.tight_layout()

            fig.savefig(os.path.join(out_dir, f"{i:04d}.png"))
            plt.close()
        
        break
