import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from extractor_sd import load_model, process_features_and_mask, get_mask
from utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
import matplotlib.pyplot as plt
import sys
from extractor_dino import ViTExtractor
from sklearn.decomposition import PCA as sklearnPCA
import math
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from cond_corr.model.correspondence_conditioner import BilinearInterpolator

class SD_DINO():
    def __init__(self, sd=True, dino=True): 
        self.MASK = False
        self.VER = "v1-5"
        self.PCA = False
        self.CO_PCA = True
        self.PCA_DIMS = [256, 256, 256]
        self.SIZE = 960  # 60*16
        self.RESOLUTION = 256
        self.EDGE_PAD = False

        if sd and dino:
            self.FUSE_DINO = 1
            self.ONLY_DINO = 0
        if dino and not sd:
            self.FUSE_DINO = 0
            self.ONLY_DINO = 1
        if sd and not dino:
            self.FUSE_DINO = 0
            self.ONLY_DINO = 0
        
        self.DINOV2 = True
        self.MODEL_SIZE = 'base' # 'small' or 'base', indicate dinov2 model
        self.TEXT_INPUT = False
        self.SEED = 42
        self.TIMESTEP = 100 #flexible from 0~200

        self.DIST = 'l2' if self.FUSE_DINO and not self.ONLY_DINO else 'cos'
        if self.ONLY_DINO:
            self.FUSE_DINO = True

        if sd:
            self.model, self.aug = load_model(diffusion_ver=self.VER, image_size=self.SIZE, num_timesteps=self.TIMESTEP)
        
        self.img_size = 840 if self.DINOV2 else 244 # 840 is 60*14
        model_dict={'small':'dinov2_vits14',
                    'base':'dinov2_vitb14',
                    'large':'dinov2_vitl14',
                    'giant':'dinov2_vitg14'}
        
        model_type = model_dict[self.MODEL_SIZE] if self.DINOV2 else 'dino_vits8'
        self.layer = 11 if self.DINOV2 else 9
        if 'l' in model_type:
            self.layer = 23
        elif 'g' in model_type:
            self.layer = 39
        self.facet = 'token' if self.DINOV2 else 'key'
        stride = 14 if self.DINOV2 else 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # indiactor = 'v2' if DINOV2 else 'v1'
        # model_size = model_type.split('vit')[-1]
        self.extractor = ViTExtractor(model_type, stride, device=self.device)
        patch_size = self.extractor.model.patch_embed.patch_size[0] if self.DINOV2 else extractor.model.patch_embed.patch_size
        self.num_patches = int(patch_size / stride * (self.img_size // patch_size - 1) + 1)

        self.interp = BilinearInterpolator()
        # self.dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').requires_grad_(False).cuda()

    def compute_pair_feature(self, img1, img2, category = [[''], ['']]):  # no categories
        mask=self.MASK
        dist=self.DIST
        real_size=self.SIZE

        if type(category) == str:
            category = [category]
        input_text = "a photo of "+category[-1][0] if self.TEXT_INPUT else None
        # input text is always none

        # Load image 1
        # img1 = Image.open(files[0]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        img1 = resize(img1, self.img_size, resize=True, to_pil=True, edge=self.EDGE_PAD)

        # Load image 2
        # img2 = Image.open(files[1]).convert('RGB')
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=self.EDGE_PAD)
        img2 = resize(img2, self.img_size, resize=True, to_pil=True, edge=self.EDGE_PAD)

        with torch.no_grad():
            if not self.CO_PCA:
                if not self.ONLY_DINO:
                    img1_desc = process_features_and_mask(self.model, self.aug, img1_input, input_text=input_text, mask=False, pca=self.PCA).reshape(1,1,-1, self.num_patches**2).permute(0,1,3,2)
                    img2_desc = process_features_and_mask(self.model, self.aug, img2_input, category[-1], input_text=input_text,  mask=mask, pca=self.PCA).reshape(1,1,-1, self.num_patches**2).permute(0,1,3,2)
                if self.FUSE_DINO:
                    img1_batch = self.extractor.preprocess_pil(img1)
                    img1_desc_dino = self.extractor.extract_descriptors(img1_batch.to(self.device), self.layer, self.facet)
                    img2_batch = self.extractor.preprocess_pil(img2)
                    img2_desc_dino = self.extractor.extract_descriptors(img2_batch.to(self.device), self.layer, self.facet)

            else:
                if not self.ONLY_DINO:
                    features1 = process_features_and_mask(self.model, self.aug, img1_input, input_text=input_text,  mask=False, raw=True)
                    features2 = process_features_and_mask(self.model, self.aug, img2_input, input_text=input_text,  mask=False, raw=True)
                    processed_features1, processed_features2 = co_pca(features1, features2, self.PCA_DIMS)
                    img1_desc = processed_features1.reshape(1, 1, -1, self.num_patches**2).permute(0,1,3,2)
                    img2_desc = processed_features2.reshape(1, 1, -1, self.num_patches**2).permute(0,1,3,2)
                if self.FUSE_DINO:
                    img1_batch = self.extractor.preprocess_pil(img1)
                    img1_desc_dino = self.extractor.extract_descriptors(img1_batch.to(self.device), self.layer, self.facet)
                    
                    img2_batch = self.extractor.preprocess_pil(img2)
                    img2_desc_dino = self.extractor.extract_descriptors(img2_batch.to(self.device), self.layer, self.facet)
                
            if dist == 'l1' or dist == 'l2':
                # normalize the features
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                if self.FUSE_DINO:
                    img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                    img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

            if self.FUSE_DINO and not self.ONLY_DINO:
                # cat two features together
                img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
                img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)

            if self.ONLY_DINO:
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino

            if self.ONLY_DINO or not self.FUSE_DINO:
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
        
        img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], self.num_patches, self.num_patches)
        mg2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], self.num_patches, self.num_patches)
        return img1_desc_reshaped, mg2_desc_reshaped
    
    def get_heatmap(self, f0_im, f1_im, f0_co):
        y, x = torch.meshgrid(torch.arange(224),torch.arange(224))
        f1_co = torch.stack([x.flatten(), y.flatten()], axis=1).unsqueeze(0)
        f1_co = f1_co.cuda()

        f0_tokens, f1_tokens = self.compute_pair_feature(f0_im, f1_im)

        f0_co = f0_co[:,:,[1,0]]
        f1_co = f1_co[:,:,[1,0]]

        f0_feat = self.interp(f0_tokens, f0_co)
        f1_feat = self.interp(f1_tokens, f1_co)

        f0_feat = torch.nn.functional.normalize(f0_feat,dim=-1)
        f1_feat = torch.nn.functional.normalize(f1_feat,dim=-1)

        sim = torch.bmm(f0_feat, f1_feat.permute(0,2,1))
        sim = sim.reshape(-1,224,224)
        if sim.shape[0] == 1:
            sim = sim.squeeze(0)
        
        return sim

if __name__ == "__main__":
    src_img_path = "<#TODO: your source image path>"
    trg_img_path = "<#TODO: your target image path>"
    files = [src_img_path, trg_img_path]
    model = SD_DINO(sd=False, dino=True)
    img1_desc, img2_desc = model.compute_pair_feature(Image.open(files[0]).convert('RGB'), Image.open(files[0]).convert('RGB'))
    print(img1_desc.shape, img2_desc.shape)