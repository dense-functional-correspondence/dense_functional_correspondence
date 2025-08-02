import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import textwrap

from PIL import Image

def normalize_tensor(tensor):
    """
    Normalize a PyTorch tensor to the range [0, 1].

    Parameters:
        tensor (torch.Tensor): The input tensor to normalize.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    # Ensure the tensor is a floating point data type for precise division
    tensor = tensor.float()
    
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    # Prevent division by zero in case the tensor is constant
    if max_val - min_val > 0:
        # Normalize the tensor
        tensor = (tensor - min_val) / (max_val - min_val)
    else:
        # If the tensor is constant, return 0s since normalization is not meaningful
        tensor = torch.zeros_like(tensor)
    
    return tensor

def convert_tensor_image(tensor, normalize=True):
    if tensor.device.type == "cuda":
        tensor = tensor.detach().cpu()

    if normalize:
        # Normalize the tensor to [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Scale to [0, 255] and convert to numpy array
    image = (tensor.numpy() * 255).astype(np.uint8)

    # Handle the case where the tensor is grayscale (1 channel)
    if image.shape[0] == 1:
        image = image.squeeze(0)

    elif image.shape[0] == 3 or image.shape[0] == 4:
        image = np.transpose(image, (1, 2, 0))

    image = Image.fromarray(image)
    return image

@torch.no_grad()
def make_viz(loader, model, out_dir, tag):

    model.eval()

    for itr, batch in enumerate(loader):
        image = batch['img'].cuda()
        max_hmap_co = batch['max_hmap_co'].cuda()
        lang_feat = batch['lang_feat'].cuda()
        
        for i in torch.arange(0,32,2):
            f0_im = image[i].unsqueeze(0)
            f1_im = image[i+1].unsqueeze(0)

            f0_co = max_hmap_co[i].unsqueeze(0).unsqueeze(0)
            y, x = torch.meshgrid(torch.arange(224),torch.arange(224))
            f1_co = torch.stack([x.flatten(), y.flatten()], axis=1).unsqueeze(0)
            f1_co = f1_co.cuda()

            f0_lang_feat = lang_feat[i].unsqueeze(0)
            f1_lang_feat = lang_feat[i+1].unsqueeze(0)

            f0_dino_tokens = model.image_embedder(f0_im)
            f1_dino_tokens = model.image_embedder(f1_im)

            f0_co = f0_co[:,:,[1,0]]
            f1_co = f1_co[:,:,[1,0]]

            f0_feat = model.mlp_conditioner(f0_dino_tokens, f0_lang_feat, f0_co)
            f1_feat = model.mlp_conditioner(f1_dino_tokens, f1_lang_feat, f1_co)

            sim = torch.bmm(f0_feat, f1_feat.permute(0,2,1))
            sim = sim.reshape(224,224)
            
            img1 = convert_tensor_image(f0_im[0])
            img2 = convert_tensor_image(f1_im[0])

            fig, axs = plt.subplots(nrows=1, ncols=3)

            for ax in axs.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            f0_co = f0_co.squeeze().cpu().numpy()

            axs[0].imshow(img1)
            axs[1].imshow(img2)
            axs[2].imshow(convert_tensor_image(sim))
            axs[0].scatter(f0_co[1], f0_co[0], s=5, c='r')

            name1 = batch['name'][i]
            name2 = batch['name'][i+1]

            axs[0].set_title(name1)
            axs[1].set_title(name2)

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{tag}_{i:04d}.png"))
            plt.close()

        break
    
    model.train()

def beautify_long_name(name):
    new_name = "||".join(name.split("||")[:2]) + "\n"
    new_name += "\n".join(textwrap.wrap(name.split("||")[2], width=24))
    return new_name

def visualize(viz_dict, out_dir):
    
    n_viz = len(list(viz_dict.values())[0]) ## how many things we'll need to visualize

    for i in range(n_viz):
        # unpack all the items
        f0 = viz_dict['f0'][i]
        f1 = viz_dict['f1'][i]
        f0_co_list = viz_dict['f0_co'][i]
        name1 = viz_dict['name1'][i]
        name2 = viz_dict['name2'][i]
        gt_hmap_query = viz_dict['gt_hmap_query'][i]
        gt_hmap = viz_dict['gt_hmap'][i]
        pred_hmap_list = viz_dict['pred_hmap'][i]
        sim_list = viz_dict['sim'][i]

        if "pred_mask0" in viz_dict:
            pred_mask0 = viz_dict['pred_mask0'][i]
        else:
            pred_mask0 = None
        if "pred_mask1" in viz_dict:
            pred_mask1 = viz_dict['pred_mask1'][i]
        else:
            pred_mask1 = None

        fig, axs = plt.subplots(nrows=len(f0_co_list), ncols=5)

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(len(f0_co_list)):
            sim_to_viz = normalize_tensor(pred_hmap_list[j]).numpy()
            
            axs[j][0].imshow(convert_tensor_image(f0))
            f0_co = np.array(f0_co_list[j])
            if len(f0_co.shape) == 1:
                axs[j][0].scatter(f0_co[0], f0_co[1], s=7.5, c='r')
            else:
                axs[j][0].scatter(f0_co[:, 0], f0_co[:, 1], s=7.5, c='r')
            if pred_mask0 is not None:
                axs[j][0].imshow(pred_mask0, alpha=0.5, cmap='plasma')

            axs[j][1].imshow(convert_tensor_image(f1))
            if pred_mask1 is not None:
                axs[j][1].imshow(pred_mask1, alpha=0.5, cmap='plasma')

            axs[j][2].imshow(convert_tensor_image(f1))
            axs[j][2].imshow(sim_to_viz, alpha=0.5, cmap='plasma')
            flat_index = np.argmax(sim_to_viz)
            row, col = np.unravel_index(flat_index, sim_to_viz.shape)
            axs[j][2].scatter(col, row, s=7.5, c='r')
            
            axs[j][3].imshow(convert_tensor_image(f0))
            axs[j][3].imshow(gt_hmap_query, alpha=0.5, cmap='plasma')

            axs[j][4].imshow(convert_tensor_image(f1))
            axs[j][4].imshow(gt_hmap, alpha=0.5, cmap='plasma')

            axs[j][0].set_title(beautify_long_name(name1), fontsize=12)
            axs[j][1].set_title(beautify_long_name(name2), fontsize=12)
            axs[j][2].set_title(f"pred hmap | sim:{sim_list[j]:.2f}", fontsize=12)
            axs[j][3].set_title("gt_query_img", fontsize=12)
            axs[j][4].set_title("gt", fontsize=12)
        
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
        # fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.set_size_inches(15,3.0*len(f0_co_list)+1.0) # 12, 3.5
        fig.savefig(os.path.join(out_dir, f"{i:04d}.png"))
        plt.close()


def visualize_spatial(viz_dict, out_dir):
    
    n_viz = len(list(viz_dict.values())[0]) ## how many things we'll need to visualize

    for i in range(n_viz):
        # unpack all the items
        f0 = viz_dict['f0'][i]
        f1 = viz_dict['f1'][i]
        f0_co = viz_dict['f0_co'][i]
        f1_co_pred = viz_dict['f1_co_pred'][i]
        f1_co = viz_dict['f1_co'][i]
        name1 = viz_dict['name1'][i]
        name2 = viz_dict['name2'][i]
        part = viz_dict['part'][i]
        pred_hmap = viz_dict['pred_hmap'][i]
        dist = viz_dict['dist'][i]

        fig, axs = plt.subplots(nrows=1, ncols=3)

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        sim_to_viz = normalize_tensor(pred_hmap).numpy()
        
        axs[0].imshow(convert_tensor_image(f0))
        f0_co = np.array(f0_co)
        if len(f0_co.shape) == 1:
            axs[0].scatter(f0_co[0], f0_co[1], s=7.5, c='r')
        else:
            axs[0].scatter(f0_co[:, 0], f0_co[:, 1], s=7.5, c='r')

        axs[1].imshow(convert_tensor_image(f1))
        axs[1].scatter(f1_co[0], f1_co[1], s=7.5, c='r')

        axs[2].imshow(convert_tensor_image(f1))
        axs[2].imshow(sim_to_viz, alpha=0.5, cmap='plasma')
        axs[2].scatter(f1_co_pred[0], f1_co_pred[1], s=7.5, c='r')

        axs[0].set_title(name1 + f" | {part}", fontsize=12)
        axs[1].set_title(name2 + " | gt", fontsize=12)
        axs[2].set_title(f"pred hmap | dist:{dist:.2f}", fontsize=12)
        
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.set_size_inches(10,4)
        fig.savefig(os.path.join(out_dir, f"{i:04d}.png"))
        plt.close()


def visualize_CLIP_baseline(viz_dict, out_dir):
    
    n_viz = len(list(viz_dict.values())[0]) ## how many things we'll need to visualize

    for i in range(n_viz):
        # unpack all the items
        image = viz_dict['image'][i]
        name = viz_dict['name'][i]
        query_idx = viz_dict['query_idx'][i]
        gt_hmap = viz_dict['gt_hmap'][i]
        pred_hmap = viz_dict['pred_hmap'][i]
        sim = viz_dict['sim'][i]

        fig, axs = plt.subplots(nrows=1, ncols=3)

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        sim_to_viz = normalize_tensor(pred_hmap).numpy()
        
        axs[0].imshow(convert_tensor_image(image))

        axs[1].imshow(convert_tensor_image(image))
        axs[1].imshow(sim_to_viz, alpha=0.5, cmap='plasma')

        axs[2].imshow(convert_tensor_image(image))
        axs[2].imshow(gt_hmap, alpha=0.5, cmap='plasma')

        axs[0].set_title(beautify_long_name(name), fontsize=12)
        axs[1].set_title(f"pred hmap | sim:{sim:.2f}", fontsize=12)
        axs[2].set_title(f"gt||query_{query_idx}", fontsize=12)
        
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.set_size_inches(9,4) # 12, 3.5
        fig.savefig(os.path.join(out_dir, f"{i:04d}.png"))
        plt.close()
