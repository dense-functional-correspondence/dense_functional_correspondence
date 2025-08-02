import torch
import torchvision
import numpy as np
import json
import random
import os
import io
import glob
import bisect

from torchvision import transforms
from tqdm import tqdm

from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from cond_corr.data.utils.aug_utils import HorizontalFlip, RandomCrop, resize_image_mask_box, pad_to_square, pad_to_square_reflective
from cond_corr.data.utils.aug_utils_dope import ContrastiveAugmentation, get_random_image
from cond_corr.data.utils.load_utils import load_data_in_ram

class BaseCogVLM_loader(Dataset):
    def __init__(self, split="", image_size=224, load_in_ram=True, mode="train", num_points=128, 
                test_selected=False, subsample=None, clip_dir=None, aug_params=None):

        assert split in ['train','test']
        assert mode in ['train', 'eval']
        assert(clip_dir is not None)

        self.test_selected = test_selected
        self.data_list = self.load_dataset(split, test_selected)
        if subsample is not None:
            print(f"Subsampled {subsample} data points from {len(self.data_list)}")
            self.data_list = random.sample(self.data_list, subsample)

        self.affordance_label_dict = self.build_label_dict(self.data_list)
        self.affordance_text_feats = self.load_aff_text_features(clip_dir)

        if load_in_ram:
            self.data_list = load_data_in_ram(self.data_list)

        self.horizontal_flipping = HorizontalFlip(1.0)
        self.random_crop = RandomCrop()
        self.image_size = image_size
        self.mode = mode
        self.split = split
        self.num_points = num_points

        if aug_params is not None:
            self.transform = ContrastiveAugmentation(aug_params)
        else:
            self.transform = None

    @staticmethod
    def load_aff_text_features(clip_dir):
        feat_path = clip_dir
        items = os.listdir(feat_path)
        feat_dict = {}
        
        for item in items:
            item_path = os.path.join(feat_path, item)
            feat_dict[item.replace('.pth','')] = torch.load(item_path)

        return feat_dict

    @staticmethod
    def build_label_dict(data_list):
        affs = sorted(list(set([x['name'].split("||")[0] for x in data_list])))
        label_dict = {x:torch.zeros(len(data_list)) for x in affs}

        for idx, item in enumerate(data_list):
            affordance_label = item['name'].split("||")[0]
            label_dict[affordance_label][idx] = 1

        keys = list(label_dict.keys())
        for k in keys:
            if sum(label_dict[k]) < 3:
                label_dict.pop(k)

        return label_dict


    @staticmethod
    def load_dataset(split):
        raise NotImplementedError
    
    @staticmethod
    def generate_coordinates_mask(bbox, mask, N, anno):
        
        mask[mask==255] = 1
        x_min, y_min, x_max, y_max = bbox
        bin_img = np.zeros_like(mask)
        bin_img[y_min:y_max, x_min:x_max] = 1

        label = mask * bin_img
        # only sample negative labels from the rest of object!
        neg_label = mask - label
        pos_idxs = np.stack(np.where(label==1)).T
        neg_idxs = np.stack(np.where(neg_label==1)).T
        if len(neg_idxs) < N:
            neg_idxs = np.stack(np.where(label==0)).T

        if len(pos_idxs) < N:  # HACK: handle some edge case
            perm = np.concatenate((np.random.permutation(len(pos_idxs)), np.random.permutation(len(pos_idxs))))[:N]
        else:
            perm = np.random.permutation(len(pos_idxs))[:N]
        pos_idxs = pos_idxs[perm]
        pos_co = pos_idxs[:,[1,0]]

        perm = np.random.permutation(len(neg_idxs))[:N]
        neg_idxs = neg_idxs[perm]
        neg_co = neg_idxs[:,[1,0]]

        return pos_co, neg_co, label
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx, bbox_idx=None):
        if isinstance(idx, tuple):
            idx, bbox_idx = idx
        anno = self.data_list[idx]
        name = anno['name']
        aff = name.split("||")[0]

        img = Image.open(anno['img_path']).convert('RGB')
        mask = Image.open(anno['mask_path']).convert('L')

        bboxes = anno['bboxes']
        if bbox_idx is not None:
            bbox = bboxes[anno['bbox_idxs'].index(bbox_idx)]
        else:
            bbox = random.choice(bboxes)

        img, mask, bbox = resize_image_mask_box(img, mask, bbox, self.image_size)
        img, mask, bbox = pad_to_square(img, mask, bbox, random_pad=self.mode=="train")

        if self.mode == "train" and self.transform is not None:
            pass # TODO, not implemented yet due to bbox...

        # segment object
        white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.composite(img.convert("RGBA"), white_bg, mask)
        img = img.convert("RGB")

        mask = np.array(mask)//255

        pos_co, neg_co, aff_label = self.generate_coordinates_mask(bbox, mask, self.num_points, anno)

        pos_co = torch.tensor(pos_co).long()
        neg_co = torch.tensor(neg_co).long()

        img = torchvision.transforms.functional.to_tensor(img)

        gt_img = torch.zeros(224,224)
        gt_img[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1 # binary "heatmap" image

        out = {}
        out['img'] = img
        out['binary_mask'] = mask
        out['gt_hmap'] = gt_img
        out['gt_hmap_mask'] = aff_label
        out['name'] = name
        out['pos_co'] = pos_co
        out['neg_co'] = neg_co
        out['max_hmap_co'] = pos_co.float().mean(0)
        out['lang_feat'] = self.affordance_text_feats[aff]
    
        return out


# For 3D data sets
class CogVLM_3D_loader(BaseCogVLM_loader):
    def __init__(self, split="", image_size=224, load_in_ram=True, mode="train", num_points=128, 
                test_selected=False, subsample=None, clip_dir=None, aug_params=None):
        super().__init__(split, image_size, load_in_ram, mode, num_points, test_selected, subsample, clip_dir, aug_params)
        
    @staticmethod
    def generate_coordinates_mask(pseudo_label, mask, N):
        mask[mask==255] = 1
        label = pseudo_label
        # only sample negative labels from the rest of object!
        neg_label = mask - label
        pos_idxs = np.stack(np.where(label==1)).T
        neg_idxs = np.stack(np.where(neg_label==1)).T

        if len(pos_idxs) == 0: # HACK: handles some edge case
            pos_idxs = np.stack(np.where(mask==0)).T
            neg_idxs = np.stack(np.where(mask==0)).T

        if len(neg_idxs) < N:
            neg_idxs = np.stack(np.where(label==0)).T
        if len(pos_idxs) < N:  # HACK: handles some edge case
            k = N // len(pos_idxs) + 1
            perm = np.concatenate([np.random.permutation(len(pos_idxs)) for _ in range(k)])[:N]
        else:
            perm = np.random.permutation(len(pos_idxs))[:N]
        pos_idxs = pos_idxs[perm]
        pos_co = pos_idxs[:,[1,0]]

        perm = np.random.permutation(len(neg_idxs))[:N]
        neg_idxs = neg_idxs[perm]
        neg_co = neg_idxs[:,[1,0]]

        return pos_co, neg_co, label

    def __getitem__(self, idx, bbox_idx=None):
        if isinstance(idx, tuple):
            idx, bbox_idx = idx
        anno = self.data_list[idx]
        name = anno['name']
        aff = name.split("||")[0]

        bboxes = anno['bboxes']
        if bbox_idx is not None:
            bbox = bboxes[anno['bbox_idxs'].index(bbox_idx)]
        else:
            bbox = random.choice(bboxes)

        img = Image.open(anno['img_path']).convert('RGB')
        mask = Image.open(anno['mask_path']).convert('L')
        pseudo_label = Image.open(bbox).convert('L')
        assert(img.size == (224,224))
        assert(mask.size == (224,224))
        assert(pseudo_label.size == (224,224))

        # segmented object
        if self.mode == "train" and self.transform is not None:
            pass
        else:
            white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.composite(img.convert('RGBA'), white_bg, mask)
            img = img.convert("RGB")
        
        pseudo_label = np.array(pseudo_label) // 255

        pos_co, neg_co, aff_label = self.generate_coordinates_mask(pseudo_label, np.array(mask) // 255, self.num_points)
        
        if self.mode == "train" and self.transform is not None:
            co = np.concatenate((pos_co, neg_co), axis=0)
            img, mask, _, co = self.transform(img.convert('RGB'), mask.convert('RGB'), None, co)
            pos_co = co[:self.num_points,:]
            neg_co = co[self.num_points:,:]
            img = img.convert("RGB")
            mask = mask.convert("L")

        mask = np.array(mask) // 255

        pos_co = torch.tensor(pos_co).long()
        neg_co = torch.tensor(neg_co).long()

        img = torchvision.transforms.functional.to_tensor(img)

        out = {}
        out['img'] = img
        out['binary_mask'] = mask
        out['gt_hmap'] = aff_label
        out['gt_hmap_mask'] = aff_label
        out['name'] = name
        out['pos_co'] = pos_co
        out['neg_co'] = neg_co
        out['max_hmap_co'] = pos_co[0]
        out['lang_feat'] = self.affordance_text_feats[aff]
        
        return out


# For multiview contrastive loss
class Spatial_loader(Dataset):
    def __init__(self, split="", image_size=224, load_in_ram=True, mode="train", num_points=128, 
                test_selected=False, subsample=None, clip_dir=None, obj2part2function_dir=None, aug_params=None, spatial_partonly=True):

        assert split in ['train','test']
        assert mode in ['train', 'eval']

        self.image_size = image_size
        self.mode = mode
        self.split = split
        self.num_points = num_points

        self.test_selected = test_selected
        self.data_list, self.pair_list = self.load_dataset(split, test_selected)
        self.num_pairs = len(self.pair_list)
        if subsample is not None:
            print("not implemented!")

        if load_in_ram:
            self.data_list = load_data_in_ram(self.data_list)
        self.affordance_text_feats = self.load_aff_text_features(clip_dir)
        self.obj2part2function = json.load(open(obj2part2function_dir))
        self.spatial_partonly = spatial_partonly

        self.horizontal_flipping = HorizontalFlip(1.0)
        self.random_crop = RandomCrop()
        if aug_params is not None:
            self.transform = ContrastiveAugmentation(aug_params)
        else:
            self.transform = None

    def load_dataset(self, split, test_selected):
        raise NotImplementedError
    
    @staticmethod
    def load_aff_text_features(clip_dir):
        feat_path = clip_dir
        items = os.listdir(feat_path)
        feat_dict = {}
        
        for item in items:
            item_path = os.path.join(feat_path, item)
            feat_dict[item.replace('.pth','')] = torch.load(item_path)

        return feat_dict

    @staticmethod
    def generate_coordinates(vis_pt_idx1, proj_2D_loc1, vis_pt_idx2, proj_2D_loc2, mask1, mask2, num_points):
        # Find the common elements and their indices
        common_elements = np.intersect1d(vis_pt_idx1, vis_pt_idx2)
        indices1 = np.where(np.isin(vis_pt_idx1, common_elements))[0]
        indices2 = np.where(np.isin(vis_pt_idx2, common_elements))[0]
        points1 = proj_2D_loc1[indices1]
        points2 = proj_2D_loc2[indices2]

        valid_mask1 = mask1[points1[:, 0], points1[:, 1]] == 1
        valid_mask2 = mask2[points2[:, 0], points2[:, 1]] == 1
        combined_valid_mask = valid_mask1 & valid_mask2
        points1 = points1[combined_valid_mask]
        points2 = points2[combined_valid_mask]

        # correspondence should be a well-defined function
        _, unique_indices = np.unique(points1, axis=0, return_index=True)
        points1 = points1[unique_indices]
        points2 = points2[unique_indices]

        assert(len(points1) == len(points2))
        if num_points <= len(points1):
            sampled_indices = np.random.choice(len(points1), num_points, replace=False)
        else:
            sampled_indices = np.random.choice(len(points1), num_points, replace=True)
        points1 = points1[sampled_indices]
        points2 = points2[sampled_indices]

        points1 = points1[:,[1,0]] # row col -> col row
        points2 = points2[:,[1,0]] # row col -> col row
        return points1, points2
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        anno1 = self.data_list[pair[0]]
        anno2 = self.data_list[pair[1]]

        name1 = anno1['name']
        name2 = anno2['name']
        common_parts = pair[2]
        part = random.choice(common_parts)
        asset = name1.split("||")[0]
        # asset_cat = " ".join(asset.split("_")[:-1])
        asset_cat = asset.split("---")[-1].replace(".glb", "")
        affordances = self.obj2part2function[asset_cat][part]
        affordance = random.choice(affordances)

        img1 = Image.open(anno1['img_path']).convert('RGB')
        mask1 = Image.open(anno1['mask_path']).convert('L')
        img2 = Image.open(anno2['img_path']).convert('RGB')
        mask2 = Image.open(anno2['mask_path']).convert('L')
        
        if self.spatial_partonly:
            # sampling from part only
            label1 = Image.open(anno1['bboxes'][anno1['bbox_idxs'].index(part)]).convert('L')
            label2 = Image.open(anno2['bboxes'][anno2['bbox_idxs'].index(part)]).convert('L')
            label1 = np.array(label1)//255
            label2 = np.array(label2)//255
        else:
            # sampling from full image
            label1 = np.array(mask1)//255
            label2 = np.array(mask2)//255
        
        co1, co2 = self.generate_coordinates(anno1["vis_pt_idx"], anno1["proj_2D_loc"], anno2["vis_pt_idx"], anno2["proj_2D_loc"], label1, label2, self.num_points)
        
        if self.mode == "train" and self.transform is not None:
            img1, mask1, _, co1 = self.transform(img1.convert('RGB'), mask1.convert('RGB'), None, co1)
            img1 = img1.convert("RGB")
            mask1 = mask1.convert("L")
            img2, mask2, _, co2 = self.transform(img2.convert('RGB'), mask2.convert('RGB'), None, co2)
            img2 = img2.convert("RGB")
            mask2 = mask2.convert("L")
        else:
            white_bg = Image.new('RGBA', img1.size, (255, 255, 255, 255))
            img1 = Image.composite(img1.convert("RGBA"), white_bg, mask1)
            img1 = img1.convert("RGB")
            img2 = Image.composite(img2.convert("RGBA"), white_bg, mask2)
            img2 = img2.convert("RGB")

        mask1 = np.array(mask1)//255
        mask2 = np.array(mask2)//255

        co1 = torch.tensor(co1).long()
        co2 = torch.tensor(co2).long()

        img1 = torchvision.transforms.functional.to_tensor(img1)
        img2 = torchvision.transforms.functional.to_tensor(img2)

        out = {}
        out['img1'] = img1
        out['binary_mask1'] = mask1
        out['name1'] = name1
        out['img2'] = img2
        out['binary_mask2'] = mask2
        out['name2'] = name2
        out['co1'] = co1
        out['co2'] = co2
        out["part"] = part
        out["affordance"] = affordance 
        out['lang_feat'] = self.affordance_text_feats[affordance]
    
        return out


class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

        self.affordance_label_dict = self.concatenate_dict_of_tensors(
            [x.affordance_label_dict for x in datasets])
        self.split = datasets[0].split

    @staticmethod
    def concatenate_dict_of_tensors(dict_list):
        # Check if the list is empty or contains only one dictionary
        if not dict_list:
            raise ValueError("The input list is empty.")
        if len(dict_list) == 1:
            return dict_list[0]
            
        # we should allow datasets with different keys to merge.
        # Extract all unique keys from all dictionaries
        all_keys = set()
        for d in dict_list:
            all_keys.update(d.keys())
        
        # missing key we need to fill with 0s
        for d in dict_list:
            for key in all_keys:
                if key not in d:  
                    d[key] = list(d.values())[0] * 0.0
            
        # Verify that all dictionaries have the same keys
        for d in dict_list:
            if set(d.keys()) != all_keys:
                raise ValueError("Not all dictionaries have the same keys.")
            vs = list(d.values())
            length = [len(v) for v in vs]
            if len(set(length)) != 1:
                raise ValueError("Not all values in the dictionary have the same length.")

        # Concatenate tensors under each key
        result = {key: torch.cat([d[key] for d in dict_list], dim=0) for key in all_keys}

        return result
    
    def __getitem__(self, idx, bbox_idx=None):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if bbox_idx is not None:
            return self.datasets[dataset_idx][sample_idx, bbox_idx]
        else:
            return self.datasets[dataset_idx][sample_idx]
