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
from cond_corr.data.base_loader import BaseCogVLM_loader, CogVLM_3D_loader, Spatial_loader

DATA_ROOT = "<#TODO: path to the dataset root>"
METADATA_PATH = "<#TODO: path to the dataset metadata>"  # e.g. "../0_taxonomy_and_metadata/Objaverse/"

class Objaverse_CC_spatial(Spatial_loader):
    def __init__(self, split="", image_size=224, load_in_ram=True, mode="train",
                num_points=128, test_selected=False, subsample=None, aug_params=None, spatial_partonly=True):
        super().__init__(
            split=split, 
            image_size=image_size, 
            load_in_ram=load_in_ram, 
            mode=mode,
            num_points=num_points,
            test_selected = test_selected,
            subsample = subsample,
            clip_dir = "<#TODO: path to CLIP text features>",
            obj2part2function_dir = "<#TODO: path to obj2part2function.json>",
            aug_params = aug_params,
            spatial_partonly = spatial_partonly)

    def load_dataset(self, split, test_selected):
        start_time = time.time()

        if test_selected:
            assert(split == "test")
            data_list = os.path.join(METADATA_PATH, f"{split}_selected_spatial_part_list.json")
        else:
            data_list = os.path.join(METADATA_PATH, f"{split}_spatial_part_list.json")
        data_list = json.load(open(data_list))
        
        # data_list = data_list[:2000] # HACK for debugging only

        asset_image_to_idx = {}
        data_dict_list = []
        pair_list = []

        print("")
        cur_asset = None
        for item in data_list:
            asset, im1, im2, common_parts = item
            if asset != cur_asset:
                cur_asset = asset
                anno_path = os.path.join(DATA_ROOT, "fully_processed_data", asset, "visible_pt_dict_processed.npz")
                assert os.path.exists(anno_path)
                visible_pt_dict = np.load(anno_path, allow_pickle=True)
                visible_pt_dict = {key: visible_pt_dict[key] for key in visible_pt_dict}

            for image in [im1, im2]:
                item_name = f"{asset}||{image}"
                img_path = os.path.join(DATA_ROOT, "fully_processed_data", asset, "rgb_images_processed", image)
                mask_path = os.path.join(DATA_ROOT, "fully_processed_data", asset, "object_masks_processed", image)
                label_path = os.path.join(DATA_ROOT, "fully_processed_data", asset, "pseudo_labels_processed")
                assert os.path.exists(img_path)
                assert os.path.exists(mask_path)
                assert os.path.exists(label_path)

                if item_name not in asset_image_to_idx:
                    asset_image_to_idx[item_name] = len(data_dict_list)
                    assert(asset == cur_asset)
                    # visible_pt_dict = np.load(anno_path, allow_pickle=True)
                    # visible_pt_dict = {key: visible_pt_dict[key] for key in visible_pt_dict}
                    img_num = image
                    visible_pt = visible_pt_dict[img_num].flatten()[0]
                    vis_pt_idx = visible_pt["vis_pt_idx"].astype("uint32")
                    proj_2D_loc = visible_pt["proj_2D_loc"].astype("uint8")

                    bbox_idxs = []
                    bboxes = []
                    for part in os.listdir(label_path):
                        bbox_idxs.append(part)
                        bbox_dir = os.path.join(label_path, part, image)
                        assert os.path.exists(bbox_dir)
                        bboxes.append(bbox_dir)

                    dct = {
                        "name":item_name,
                        "img_path":img_path,
                        "mask_path":mask_path,
                        "vis_pt_idx":vis_pt_idx,
                        "proj_2D_loc":proj_2D_loc,
                        "bboxes":bboxes,
                        "bbox_idxs":bbox_idxs
                        }

                    data_dict_list.append(dct)
                    print(f"Building data list: {len(data_dict_list):06d}", end='\r')
            
            pair_list.append([asset_image_to_idx[f"{asset}||{im1}"],asset_image_to_idx[f"{asset}||{im2}"], common_parts])

        print(f"loaded {len(data_dict_list)} asset-view combinations and {len(pair_list)} pairs.")
        print(f"Loading data set took {time.time()-start_time:.2f} seconds.")
        return data_dict_list, pair_list
    
    @staticmethod
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

        new_x0 = max(int(new_x0), 0)
        new_y0 = max(int(new_y0), 0)
        new_x1 = min(int(new_x1), 490)
        new_y1 = min(int(new_y1), 490)
        
        return [new_x0, new_y0, new_x1, new_y1]
    

if __name__ == "__main__":

    SEED = 1234     
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    import matplotlib.pyplot as plt
    from cond_corr.data.sampler import SpatialBatchSampler

    split = "train"
    n_iter = 100
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
        "RandomMaskingAug": [1.0, 0.5,"domain_randomized", ["rotation", "scaling"]]
    }
    dataset = Objaverse_CC_spatial(split=split, load_in_ram=True, mode="train", test_selected=False, aug_params=aug_params, spatial_partonly=False)
    sampler = SpatialBatchSampler(dataset.num_pairs, n_iter, batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8)
    sampler.rng.seed(SEED)

    out_dir = f"<#TODO: your dir>/experiment_logs/data_samples/objaverse_spatialLoader_fullImg_outputs_{split}"
    os.makedirs(out_dir, exist_ok=True)

    for batch in loader:
        n_items = len(batch['img1'])
        for i in range(0,n_items):
            print(i)

            img1 = batch['img1'][i]
            img2 = batch['img2'][i]

            co1 = batch['co1'][i]
            co2 = batch['co2'][i]

            name1 = batch['name1'][i]
            name2 = batch['name2'][i]

            part = batch['part'][i]
            affordance = batch['affordance'][i]

            print(name1, name2)

            img1 = np.array(torchvision.transforms.functional.to_pil_image(img1))
            img2 = np.array(torchvision.transforms.functional.to_pil_image(img2))
            img = np.concatenate((img1, img2), axis=1)
            plt.figure()
            plt.imshow(img)
            cmap = plt.get_cmap('jet')
            H0, W0, H1, W1 = *img1.shape[:2], *img2.shape[:2]
            
            n_viz_points = 20
            co1, co2 = co1[:n_viz_points, :], co2[:n_viz_points, :]
            sorted_indices = np.argsort(co1[:, 1])
            co1 = co1[sorted_indices]
            co2 = co2[sorted_indices]

            for j, ((x1, y1), (x2, y2)) in enumerate(zip(co1, co2)):
                plt.plot([x1, x2 + W0], [y1, y2], '-+', color=cmap(j / (n_viz_points-1)), scalex=False, scaley=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{i:04d}-{affordance}-{part}.png"))
            plt.close()
        
        break
