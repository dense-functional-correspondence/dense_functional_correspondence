import torch
import torch.nn as nn
from torchvision import transforms
# from featup.util import norm, unnorm
import torch.nn.modules.utils as nn_utils
import torch.nn.functional as F
from typing import Union, List, Tuple
import types
import math
import numpy as np

from einops.layers.torch import Rearrange
import loralib as lora

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, text_tokens, vision_tokens):

        tgt = text_tokens
        tgt2 = self.multihead_attn(query=text_tokens,
                                   key=vision_tokens,
                                   value=vision_tokens)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DINO_embedder(nn.Module):
    def __init__(self, use_last_layers=3, use_smaller_stride=False, dino_size="base"):
        super().__init__()

        self.n_blocks = use_last_layers
        print(f"{dino_size} DINO embedder is using last {self.n_blocks} blocks.\n")

        if dino_size == "base":
            self.dino_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').requires_grad_(False)
        elif dino_size == "small":
            self.dino_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').requires_grad_(False)
        elif dino_size == "large":
            self.dino_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').requires_grad_(False)
        elif dino_size == "giant":
            self.dino_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').requires_grad_(False)

        # different stride, higher map resolution, seems to be quadratically slower?
        if use_smaller_stride:
            self.dino_vit = DINO_embedder.patch_vit_resolution(self.dino_vit, stride=7)
            print(f"DINO embedder is using a smaller stride.\n")
        self.image_embed_dim = 512
        self.lin_proj = nn.Linear(self.dino_vit.num_features, self.image_embed_dim, bias=False)
        self.w = nn.Parameter(torch.rand(self.n_blocks))
        self.imagenet_normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def get_DINO_feats(self, x, n_blocks=3):
        x = self.imagenet_normalization(x)
        dino_features = self.dino_vit.get_intermediate_layers(x, n=n_blocks, norm=False)

        # B, n_blocks, n_patches, C
        dino_features = torch.stack(dino_features,dim=1)
        w = torch.ones(n_blocks).softmax(-1).reshape(1, n_blocks, 1, 1).to(dino_features.device)
        dino_features = dino_features * w
        # summing along n_blocks axis: B, n_blocks, n_patches, C -> B,  n_patches, C
        dino_features = dino_features.sum(1)
        dino_features = torch.nn.functional.normalize(dino_features, dim=-1)
        return dino_features

    def forward(self, x):
        x = self.imagenet_normalization(x)
        dino_features = self.dino_vit.get_intermediate_layers(x, n=self.n_blocks, norm=False)
        # B, n_blocks, n_patches, C
        dino_features = torch.stack(dino_features,dim=1)

        B, n_blocks, n_patches, C = dino_features.shape

        # projection
        dino_features = self.lin_proj(dino_features)
        # weighting
        w = self.w.softmax(-1).reshape(1,self.n_blocks, 1, 1)
        dino_features = dino_features * w
        # summing along n_blocks axis: B, n_blocks, n_patches, C -> B,  n_patches, C
        dino_features = dino_features.sum(1)
        dino_features = torch.nn.functional.normalize(dino_features, dim=-1)

        return dino_features
    
    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    # https://github.com/Junyi42/sd-dino/blob/f4f3ac7a52128c2688e2111b4b451f3a47797c85/extractor_dino.py#L123
    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(DINO_embedder._fix_pos_enc(patch_size, stride), model)
        return model

class FiLMLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # clip embedding is 512 dimensional
        self.gamma_layer = nn.Linear(512, embed_dim, bias=False)
        self.beta_layer = nn.Linear(512, embed_dim, bias=False)

    def forward(self, x, conditioning):
        conditioning = conditioning.float()
        gamma = self.gamma_layer(conditioning)
        beta = self.beta_layer(conditioning)
        n_pts = x.shape[1]
        gamma = gamma.unsqueeze(1).expand(-1,n_pts,-1)
        beta = beta.unsqueeze(1).expand(-1,n_pts,-1)
        return (1 + gamma) * x + beta  # (1+gamma) is crucial!!!

class DINO_B_LORA_FiLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args  # use_lora, use_film, lora_rank, n_blocks

        # Load DINOv2 model and apply LoRA to all QKV layers
        if "dino_size" in args:
            if args["dino_size"] == "small":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            elif args["dino_size"] == "big":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            elif args["dino_size"] == "large":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            elif args["dino_size"] == "giant":
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        if args["use_lora"]:
            for i in range(len(self.model.blocks)):
                module = self.model.blocks[i].attn.qkv
                new_module = lora.MergedLinear(
                    module.in_features, 
                    module.out_features, 
                    r=args["lora_rank"], 
                    enable_lora=[True, True, True],  # Enable LoRA on Q, K, and V
                    bias=module.bias is not None
                )
                new_module.load_state_dict(module.state_dict(), strict=False)
                self.model.blocks[i].attn.qkv = new_module
        lora.mark_only_lora_as_trainable(self.model)

        # Create film layers
        if args["use_film"]:
            self.film = nn.ModuleList([
                FiLMLayer(embed_dim = self.model.embed_dim) for _ in range(len(self.model.blocks))
            ])
        # print(self.model.embed_dim, self.model.num_features) both 768

        self.image_embed_dim = 512
        self.lin_proj = nn.Linear(self.model.num_features, self.image_embed_dim, bias=False)
        self.w = nn.Parameter(torch.rand(args["n_blocks"]))
        self.imagenet_normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x, clip_embedding = None):
        if self.args["use_film"]:
            assert(clip_embedding is not None)

        x = self.imagenet_normalization(x)
        x = self.model.prepare_tokens_with_masks(x)  # Patch embedding
        output, total_block_len = [], len(self.model.blocks)
        blocks_to_take = range(total_block_len - self.args["n_blocks"], total_block_len) if isinstance(self.args["n_blocks"], int) else self.args["n_blocks"]
        for i, blk in enumerate(self.model.blocks):
            if self.args["use_film"]:
                x = self.film[i](x, clip_embedding)
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        output = [out[:, 1 + self.model.num_register_tokens :] for out in output]

        # B, n_blocks, n_patches, C
        dino_features = torch.stack(output, dim=1)
        B, n_blocks, n_patches, C = dino_features.shape

        # projection
        dino_features = self.lin_proj(dino_features)
        # weighting
        w = self.w.softmax(-1).reshape(1,n_blocks, 1, 1)
        dino_features = dino_features * w
        # summing along n_blocks axis: B, n_blocks, n_patches, C -> B,  n_patches, C
        dino_features = dino_features.sum(1)
        dino_features = torch.nn.functional.normalize(dino_features, dim=-1)
        return dino_features


class DINO_FeatUp_embedder(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 384
        self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).requires_grad_(False)
        self.image_embed_dim = 256
        self.lin_proj = nn.Linear(self.channels, self.image_embed_dim, bias=False)

    def forward(self, x):
        # B, C, width, height
        dino_features = self.upsampler(norm(x))
        dino_features = nn.functional.interpolate(dino_features, size=(224,224), mode='bilinear')
        B, C, width, height = dino_features.shape
        dino_features = dino_features.reshape(B, -1, C)
        # projection, B, n_patches, C
        dino_features = self.lin_proj(dino_features)

        return dino_features

class BilinearInterpolator(nn.Module):
    def __init__(self, im_size=224):
        super(BilinearInterpolator, self).__init__()
        self.im_size = im_size

    @staticmethod
    def map_to_range(x, in_min, in_max, out_min, out_max):
        x = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return x

    def forward(self, f, uv):
        """
        f: B x C x 56 x 56 input features 1
        uv: B x N_pts x 2 pixel coordinates to extract features from f

        outputs extracted features from q and k of shape B, N_pts, C
        """
        
        uv = uv[:,:,[1,0]]

        uv = self.map_to_range(uv.float(), 0, self.im_size, -1, 1)
        uv = uv.unsqueeze(1)
        
        # this outputs B, C, 1, n_pts, mode could be 'nearest' for viz
        f_out = torch.nn.functional.grid_sample(f, uv, mode='bilinear', align_corners=False)
        # reshaping it to B, n_pts, C
        f_out = f_out.permute(0,3,1,2).squeeze(3).contiguous()

        return f_out


class MLPConditioner(nn.Module):
    def __init__(self, image_embed_dim, clip_embed_dim=512, hidden_emb_dim=1024):
        super().__init__()

        print(f"\nUsing hidden embedding dimension {hidden_emb_dim}!")
        self.mlp = nn.Sequential(
            nn.Linear(image_embed_dim+clip_embed_dim, hidden_emb_dim),
            nn.ReLU(),
            nn.Linear(hidden_emb_dim, hidden_emb_dim),
            nn.ReLU(),
            nn.Linear(hidden_emb_dim, 256, bias=False)
        )

        self.interp = BilinearInterpolator()

    def forward(self, vision_tokens, language_tokens, pt_co_2d):
        
        B, n_patch, C = vision_tokens.shape
        grid_dim = int(math.sqrt(n_patch))
        assert(grid_dim * grid_dim == n_patch)
        vision_token_grid = vision_tokens.permute(0,2,1).reshape(B, C, grid_dim, grid_dim)
        pt_vision_feats = self.interp(vision_token_grid, pt_co_2d)

        if language_tokens is not None:
            n_pts = pt_co_2d.shape[1]
            language_feats = language_tokens.unsqueeze(1).expand(-1,n_pts,-1)
            fused_feats = torch.cat([pt_vision_feats, language_feats], dim=-1) 
        else:
            fused_feats = pt_vision_feats
        
        conditioned_feats = self.mlp(fused_feats)
        conditioned_feats = torch.nn.functional.normalize(conditioned_feats,dim=-1)
        
        return conditioned_feats, pt_vision_feats


class ConditionedDino(nn.Module):
    def __init__(self, use_last_layers=3, use_smaller_stride=False, use_featup=False, negneg_loss=False, 
                mask_loss=False, additional_model_args=None):
        super().__init__()

        dino_size = "base"
        if additional_model_args is not None:
            if "dino_size" in additional_model_args:
                dino_size = additional_model_args["dino_size"]
        if additional_model_args is not None:
            if additional_model_args["use_lora"] or additional_model_args["use_film"]:
                print(f"\nUsing improved DINO with configs: {additional_model_args}")
                self.image_embedder = DINO_B_LORA_FiLM(args = additional_model_args)
                self.use_film = additional_model_args["use_film"]
            else:
                additional_model_args = None
        if additional_model_args is None:
            self.use_film = False
            if use_featup:
                print("\nUsing FeatUp to get image embeddings.\n")
                self.image_embedder = DINO_FeatUp_embedder()
            else:
                print("\nUsing DINO to get image embeddings.")
                self.image_embedder = DINO_embedder(use_last_layers, use_smaller_stride, dino_size=dino_size)
        
        if self.use_film:
            self.mlp_conditioner = MLPConditioner(self.image_embedder.image_embed_dim, clip_embed_dim=0)
        else:
            self.mlp_conditioner = MLPConditioner(self.image_embedder.image_embed_dim)

        self.T_spatial = 0.1
        self.T_semantic = 0.1
        print(f"Using temperature {self.T_semantic} for semantic and {self.T_spatial} for spatial.\n")
        self.negneg_loss = negneg_loss

        self.mask_loss = mask_loss
        if self.mask_loss:
            self.mask_projection = torch.nn.Linear(256, 1)

    def get_embedding(self, image, lang_tokens, co):
        co = co[:,:,[1,0]] # col row -> row col
        if self.use_film:
            dino_tokens = self.image_embedder(image, lang_tokens)
            conditioned_features, _ = self.mlp_conditioner(dino_tokens, None, co)
        else:
            dino_tokens = self.image_embedder(image)
            conditioned_features, _ = self.mlp_conditioner(dino_tokens, lang_tokens, co)
        return conditioned_features
    
    def get_smooth_embedding(self, image, lang_tokens, co):
        # HACK: assume batch_size = 1
        if self.use_film:
            dino_tokens = self.image_embedder(image, lang_tokens)
        else:
            dino_tokens = self.image_embedder(image)
        y, x = torch.meshgrid(torch.arange(224),torch.arange(224))
        full_co = torch.stack([x.flatten(), y.flatten()], axis=1).unsqueeze(0)
        full_co = full_co.cuda()
        full_co = full_co[:,:,[1,0]]
        if self.use_film:
            conditioned_features, _ = self.mlp_conditioner(dino_tokens, None, full_co)
        else:
            conditioned_features, _ = self.mlp_conditioner(dino_tokens, lang_tokens, full_co)
        B, n_patch, C = conditioned_features.shape
        grid_dim = int(math.sqrt(n_patch))
        assert(grid_dim * grid_dim == n_patch)
        conditioned_features = conditioned_features.permute(0,2,1).reshape(B, C, grid_dim, grid_dim)
        pooled_feature_matrix = torch.nn.functional.avg_pool2d(conditioned_features, kernel_size=5, stride=1, padding=2)
        pooled_feature_matrix = pooled_feature_matrix.squeeze(0)
        co = co[:,:,[1,0]].squeeze(0) # col row -> row col
        features_at_locations = pooled_feature_matrix[:, co[:,0], co[:,1]].T
        return features_at_locations

    def get_mask(self, image, lang_tokens):
        # HACK: assume batch size = 1
        y, x = torch.meshgrid(torch.arange(224),torch.arange(224))
        co = torch.stack([x.flatten(), y.flatten()], axis=1).unsqueeze(0)
        co = co.cuda()
        co = co[:,:,[1,0]]
        if self.use_film:
            dino_tokens = self.image_embedder(image, lang_tokens)
            feat, _ = self.mlp_conditioner(dino_tokens, None, co)
        else:
            dino_tokens = self.image_embedder(image)
            feat, _ = self.mlp_conditioner(dino_tokens, lang_tokens, co)

        logits = self.mask_projection(feat).squeeze(-1).squeeze(0)
        mask = torch.sigmoid(logits)
        threshold = 0.5
        mask[mask >= threshold] = 1.0
        mask[mask < threshold] = 0.0
        mask = mask.reshape(224,224)
        return mask

    def get_heatmap(self, f0_im, f1_im, f0_co, f0_lang_feat, f1_lang_feat):

        y, x = torch.meshgrid(torch.arange(224),torch.arange(224))
        f1_co = torch.stack([x.flatten(), y.flatten()], axis=1).unsqueeze(0)
        f1_co = f1_co.cuda()
        f0_co = f0_co[:,:,[1,0]]
        f1_co = f1_co[:,:,[1,0]]

        if self.use_film:
            f0_dino_tokens = self.image_embedder(f0_im, f0_lang_feat)
            f1_dino_tokens = self.image_embedder(f1_im, f1_lang_feat)
            f0_feat, _ = self.mlp_conditioner(f0_dino_tokens, None, f0_co)
            f1_feat, _ = self.mlp_conditioner(f1_dino_tokens, None, f1_co)
        else:
            f0_dino_tokens = self.image_embedder(f0_im)
            f1_dino_tokens = self.image_embedder(f1_im)
            f0_feat, _ = self.mlp_conditioner(f0_dino_tokens, f0_lang_feat, f0_co)
            f1_feat, _ = self.mlp_conditioner(f1_dino_tokens, f1_lang_feat, f1_co)

        sim = torch.bmm(f0_feat, f1_feat.permute(0,2,1))
        sim = sim.reshape(-1,224,224)
        if sim.shape[0] == 1:
            sim = sim.squeeze(0)
        
        return sim
    
    def get_DINO_heatmap(self, f0_im, f1_im, f0_co, n_blocks=3):

        y, x = torch.meshgrid(torch.arange(224),torch.arange(224))
        f1_co = torch.stack([x.flatten(), y.flatten()], axis=1).unsqueeze(0)
        f1_co = f1_co.cuda()

        f0_dino_tokens = self.image_embedder.get_DINO_feats(f0_im, n_blocks=n_blocks)
        f1_dino_tokens = self.image_embedder.get_DINO_feats(f1_im, n_blocks=n_blocks)

        f0_co = f0_co[:,:,[1,0]]
        f1_co = f1_co[:,:,[1,0]]

        B, n_patch, C = f0_dino_tokens.shape
        grid_dim = int(math.sqrt(n_patch))
        assert(grid_dim * grid_dim == n_patch)
        
        f0_dino_tokens = f0_dino_tokens.permute(0,2,1).reshape(B, C, grid_dim, grid_dim)
        f1_dino_tokens = f1_dino_tokens.permute(0,2,1).reshape(B, C, grid_dim, grid_dim)
        f0_feat = self.mlp_conditioner.interp(f0_dino_tokens, f0_co)
        f1_feat = self.mlp_conditioner.interp(f1_dino_tokens, f1_co)

        f0_feat = torch.nn.functional.normalize(f0_feat,dim=-1)
        f1_feat = torch.nn.functional.normalize(f1_feat,dim=-1)

        sim = torch.bmm(f0_feat, f1_feat.permute(0,2,1))
        sim = sim.reshape(-1,224,224)
        if sim.shape[0] == 1:
            sim = sim.squeeze(0)
        
        return sim

    def spatial_forward(self, image1, image2, co1, co2, lang_tokens=None):
        B = len(image1)
        co1 = co1[:,:,[1,0]] # col row -> row col
        co2 = co2[:,:,[1,0]] # col row -> row col
        if self.use_film:
            dino_tokens1 = self.image_embedder(image1, lang_tokens)
            dino_tokens2 = self.image_embedder(image2, lang_tokens)
        else:
            dino_tokens1 = self.image_embedder(image1)
            dino_tokens2 = self.image_embedder(image2)

        if lang_tokens is None:
            print("WARNING: not using language conditioning for spatial!!")
            lang_tokens = torch.zeros((B, 512), device=dino_tokens1.device)
        
        if self.use_film:
            conditioned_features1, _  = self.mlp_conditioner(dino_tokens1, None, co1)
            conditioned_features2, _ = self.mlp_conditioner(dino_tokens2, None, co2)
        else:
            conditioned_features1, _  = self.mlp_conditioner(dino_tokens1, lang_tokens, co1)
            conditioned_features2, _ = self.mlp_conditioner(dino_tokens2, lang_tokens, co2)

        similarity_matrix = torch.bmm(conditioned_features1, conditioned_features2.permute(0,2,1))
        # get l_pos and l_neg from similarity matrix
        l_pos = similarity_matrix.diagonal(dim1=-2, dim2=-1)
        l_pos = l_pos.unsqueeze(-1)
        B, n, _ = similarity_matrix.shape
        non_diagonal_mask = ~torch.eye(n, dtype=torch.bool, device=similarity_matrix.device).unsqueeze(0).expand(B, -1, -1)  # Shape (B, n, n)
        l_neg = similarity_matrix[non_diagonal_mask]

        l_pos = l_pos.reshape(-1,1)
        l_neg = l_neg.reshape(-1,n-1)

        pos_val = l_pos.detach().mean()
        neg_val = l_neg.detach().mean()
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = torch.div(logits, self.T_spatial)

        return logits, pos_val, neg_val

    def forward(self, image, lang_tokens, pos_co, neg_co, sample_all_pairs=True):
        
        B = len(image) 
        n_pos = pos_co.shape[1]
        n_neg = neg_co.shape[1]
        n_pairs = B//2
        co = torch.cat([pos_co, neg_co], dim=1)
        co = co[:,:,[1,0]] # col row -> row col

        if self.use_film:
            dino_tokens = self.image_embedder(image, lang_tokens)
            conditioned_features, _ = self.mlp_conditioner(dino_tokens, None, co)
        else:
            dino_tokens = self.image_embedder(image)
            # print(lang_tokens.shape) # 100, 512
            conditioned_features, _ = self.mlp_conditioner(dino_tokens, lang_tokens, co)

        f0 = conditioned_features[::2]
        f1 = conditioned_features[1::2]

        f0_fg_feat = f0[:, :n_pos, :]
        f0_bg_feat = f0[:, n_pos:, :]

        f1_fg_feat = f1[:, :n_pos, :]
        f1_bg_feat = f1[:, n_pos:, :] # B * num_points * C 

        l_pos = torch.bmm(f0_fg_feat, f1_fg_feat.permute(0,2,1))
        l_neg_f0_f1 = torch.bmm(f0_fg_feat, f1_bg_feat.permute(0,2,1))
        l_neg_f1_f0 = torch.bmm(f1_fg_feat, f0_bg_feat.permute(0,2,1))

        if self.negneg_loss:
            l_neg_bg = torch.bmm(f0_bg_feat, f1_bg_feat.permute(0,2,1))
        
        if sample_all_pairs:
            l_pos = l_pos.reshape(-1,1)
            l_neg_f0_f1 = l_neg_f0_f1.unsqueeze(1).expand(n_pairs, n_pos, n_pos, n_neg)
            l_neg_f1_f0 = l_neg_f1_f0.unsqueeze(1).expand(n_pairs, n_pos, n_pos, n_neg)
            l_neg_f0_f1 = l_neg_f0_f1.reshape(-1,n_neg)
            l_neg_f1_f0 = l_neg_f1_f0.reshape(-1,n_neg)

            if self.negneg_loss:
                l_neg_bg = l_neg_bg.unsqueeze(1).expand(n_pairs, n_pos, n_neg, n_neg)
                l_neg_bg = l_neg_bg.reshape(-1,n_neg)
        else:
            print("WARNING: only sampling one positive pair for semantic!")
            # sample one positive pair
            l_pos = torch.diagonal(l_pos, dim1=1, dim2=2)
            l_pos = l_pos.reshape(-1,1)
            l_neg_f0_f1 = l_neg_f0_f1.reshape(-1,n_neg)
            l_neg_f1_f0 = l_neg_f1_f0.reshape(-1,n_neg)

        if self.negneg_loss:
            l_neg = torch.cat([l_neg_f0_f1, l_neg_f1_f0, l_neg_bg], dim=1)
        else:
            l_neg = torch.cat([l_neg_f0_f1, l_neg_f1_f0], dim=1)
        
        pos_val = l_pos.detach().mean()
        neg_val = l_neg.detach().mean()

        # print(l_pos.shape, l_neg.shape) # [819200, 1]; [819200, 256] 50*128*128
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = torch.div(logits, self.T_semantic)
        
        if self.mask_loss:
            f0_fg_logit = self.mask_projection(f0_fg_feat).squeeze(-1)  # B x n_pos
            f1_fg_logit = self.mask_projection(f1_fg_feat).squeeze(-1)  # B x n_pos
            f0_bg_logit = self.mask_projection(f0_bg_feat).squeeze(-1)  # B x n_neg
            f1_bg_logit = self.mask_projection(f1_bg_feat).squeeze(-1)  # B x n_neg

            # Create ground truth labels for mask loss
            pos_label = torch.ones_like(f0_fg_logit).cuda()
            neg_label = torch.zeros_like(f0_bg_logit).cuda()

            # Compute binary cross-entropy loss for mask prediction
            mask_loss = (
                F.binary_cross_entropy_with_logits(f0_fg_logit, pos_label) +
                F.binary_cross_entropy_with_logits(f1_fg_logit, pos_label) +
                F.binary_cross_entropy_with_logits(f0_bg_logit, neg_label) +
                F.binary_cross_entropy_with_logits(f1_bg_logit, neg_label)
            ) / 4
        else:
            mask_loss = 0.0

        return logits, pos_val, neg_val, mask_loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()

        loss = self.criterion(x, label)
        return loss


if __name__ == "__main__":
    '''
    dino = DINO_wrap()
    dino = dino.cuda()
    in_tensor = torch.randn(10,3,224,224).cuda()
    output = dino(in_tensor)
    '''

    vision_tokens = torch.randn(10, 256, 768)
    language_tokens = torch.randn(10, 36, 768)
    
    model = MLPConditioner()
    
    pt_co_2d = torch.randint(0,224,(10,64,2))
    pt_w_idx = torch.randint(0,36,(10,64))
    model(language_tokens, vision_tokens, pt_co_2d, pt_w_idx)

    import pdb; pdb.set_trace()
