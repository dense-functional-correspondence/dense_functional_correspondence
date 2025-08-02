import os
import random
import time
import sys

import torch
import numpy as np
import wandb

from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict

from cond_corr.utils import options
from cond_corr.data.objaverse_loader_cogvlm import Objaverse_CC_cogvlm
from cond_corr.data.objaverse_loader_spatial import Objaverse_CC_spatial
from cond_corr.data.base_loader import CustomConcatDataset
from cond_corr.data.sampler import BatchSampler, SpatialBatchSampler
from cond_corr.model.model_utils import count_parameters
from cond_corr.model.correspondence_conditioner import ConditionedDino, NCESoftmaxLoss
from cond_corr.utils.viz_utils import visualize, normalize_tensor, visualize_spatial
from cond_corr.utils.evaluation import cal_sim

def wandb_init(exp_name):
    run = wandb.init(
        entity="conditional_correspondence",
        project="cond_corr", name=exp_name)
    
    run.define_metric("epoch", hidden=True) # don't create a plot for "epoch"
    run.define_metric("iter", hidden=True) # don't create a plot for "epoch"
    run.define_metric("iter/batch_loss", step_metric="iter")
    run.define_metric("iter/pos_l", step_metric="iter")
    run.define_metric("iter/neg_l", step_metric="iter")

    run.define_metric("epoch/pos_l_train", step_metric="epoch")
    run.define_metric("epoch/neg_l_train", step_metric="epoch")
    run.define_metric("epoch/loss_train", step_metric="epoch")
    run.define_metric("epoch/sim_train", step_metric="epoch")

    run.define_metric("epoch/pos_l_test", step_metric="epoch")
    run.define_metric("epoch/neg_l_test", step_metric="epoch")
    run.define_metric("epoch/loss_test", step_metric="epoch")
    run.define_metric("epoch/sim_test", step_metric="epoch")


def train_step(model, optimizer, itr, accumulation_steps, batch, batch_spatial, loss_fn, loader_length, weight_spatial, mask_loss_weight):
    if batch is not None:
        image = batch['img'].cuda()
        # import pdb; pdb.set_trace()
        pos_co = batch['pos_co'].cuda()
        neg_co = batch['neg_co'].cuda()
        lang_feat = batch['lang_feat'].cuda()
        
        logits, pos_l_semantic, neg_l_semantic, mask_loss = model(image, lang_feat, pos_co, neg_co)
        loss_semantic = loss_fn(logits)
    else:
        loss_semantic, pos_l_semantic, neg_l_semantic, mask_loss = 0.0, 0.0, 0.0, 0.0

    if batch_spatial is not None:
        image1 = batch_spatial['img1'].cuda()
        image2 = batch_spatial['img2'].cuda()
        co1 = batch_spatial['co1'].cuda()
        co2 = batch_spatial['co2'].cuda()
        lang_feat = batch_spatial['lang_feat'].cuda()
        
        logits, pos_l_spatial, neg_l_spatial = model.spatial_forward(image1, image2, co1, co2, lang_feat)
        loss_spatial = loss_fn(logits)
    else:
        loss_spatial, pos_l_spatial, neg_l_spatial = 0.0, 0.0, 0.0

    loss = loss_semantic + mask_loss_weight * mask_loss + weight_spatial * loss_spatial
    if batch is not None and batch_spatial is not None:
        loss = loss / (1.0 + mask_loss_weight + weight_spatial)
    else:
        loss = loss / (1.0 + mask_loss_weight)
    
    # Normalize loss by the number of accumulation steps
    loss = loss / accumulation_steps
    loss.backward()

    if (itr + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    if wandb.run is not None: 
        wandb.log({"iter/loss_semantic":loss_semantic, "iter":GLOBAL_ITR}) 
        wandb.log({"iter/pos_l_semantic":pos_l_semantic, "iter":GLOBAL_ITR})
        wandb.log({"iter/neg_l_semantic":neg_l_semantic, "iter":GLOBAL_ITR})
        wandb.log({"iter/loss_spatial":loss_spatial, "iter":GLOBAL_ITR}) 
        wandb.log({"iter/pos_l_spatial":pos_l_spatial, "iter":GLOBAL_ITR})
        wandb.log({"iter/neg_l_spatial":neg_l_spatial, "iter":GLOBAL_ITR})
        wandb.log({"iter/mask_loss":mask_loss, "iter":GLOBAL_ITR})
        wandb.log({"iter/loss":loss * accumulation_steps, "iter":GLOBAL_ITR}) 
        
    if itr % 10 == 0:
        print(f"{itr:04d}/{loader_length:04d} loss:{loss.item() * accumulation_steps:.2f} pos_l_semantic:{pos_l_semantic:.3f} neg_l_semantic:{neg_l_semantic:.3f} pos_l_spatial:{pos_l_spatial:.3f} neg_l_spatial:{neg_l_spatial:.3f} mask_loss:{mask_loss:.3f}")

def train_epoch(model, optimizer, loader, loader_spatial, loss_fn, weight_spatial, mask_loss_weight):

    global GLOBAL_ITR

    # gradient accumulation -- experimental
    accumulation_steps = 1

    model.train()
    if loader is not None and loader_spatial is None:
        for itr, batch in enumerate(loader):
            batch_spatial = None
            train_step(model, optimizer, itr, accumulation_steps, batch, batch_spatial, loss_fn, len(loader), weight_spatial, mask_loss_weight)        
            GLOBAL_ITR+=1

    if loader is None and loader_spatial is not None:
        for itr, batch_spatial in enumerate(loader_spatial):
            batch = None
            train_step(model, optimizer, itr, accumulation_steps, batch, batch_spatial, loss_fn, len(loader_spatial), weight_spatial, mask_loss_weight)        
            GLOBAL_ITR+=1

    if loader is not None and loader_spatial is not None:
        for itr, (batch, batch_spatial) in enumerate(zip(loader, loader_spatial)):
            train_step(model, optimizer, itr, accumulation_steps, batch, batch_spatial, loss_fn, len(loader), weight_spatial, mask_loss_weight)        
            GLOBAL_ITR+=1
    
    # Handle remaining gradients if any
    if (itr + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

@torch.no_grad()
def eval_step(model, itr, batch, batch_spatial, loss_fn, n_viz, n_to_viz, metric_dict, viz_dict, loader_length, split):
    if batch is not None:
        start = time.time()
        # unpack the batch
        image = batch['img'].cuda()
        mask = batch['binary_mask']
        gt_hmap = batch['gt_hmap_mask']
        max_hmap_co = batch['max_hmap_co'].cuda()
        pos_co = batch['pos_co'].cuda()
        neg_co = batch['neg_co'].cuda()
        lang_feat = batch['lang_feat'].cuda()

        # get basic metrics
        logits, pos_l, neg_l, mask_loss = model(image, lang_feat, pos_co, neg_co)
        loss = loss_fn(logits)
        
        # compute metrics and gather stuff for viz
        # for i in torch.arange(0, len(image), 2):
        for i in torch.arange(0, 4, 2):  # HACK only calculate heatmap similarity for first 2 pairs
            f0_im = image[i].unsqueeze(0)
            f1_im = image[i+1].unsqueeze(0)

            name1 = batch['name'][i]
            name2 = batch['name'][i+1]
            
            f0_lang_feat = lang_feat[i].unsqueeze(0)
            f1_lang_feat = lang_feat[i+1].unsqueeze(0)

            n_query_pts = 2
            f0_co = pos_co[i][torch.randperm(len(pos_co[i]))[:n_query_pts]].unsqueeze(0)
            gt_map_query = gt_hmap[i].numpy()
            gt_map = gt_hmap[i+1].numpy()

            pred_maps = model.get_heatmap(f0_im, f1_im, f0_co, f0_lang_feat, f1_lang_feat)
            pred_maps = pred_maps.cpu()

            if model.mask_loss:
                pred_mask0 = model.get_mask(f0_im, f0_lang_feat)
                pred_mask0 = pred_mask0.cpu()
                pred_mask1 = model.get_mask(f1_im, f1_lang_feat)
                pred_mask1 = pred_mask1.cpu()

            for n_query in range(n_query_pts):
                pred_map = pred_maps[n_query]
                pred_map *= mask[i+1]
                pred_map[pred_map < 0.0] = 0.0
                pred_map += torch.min(pred_map+((1-mask[i+1])*100)) * (1 - mask[i+1])
                pred_map = normalize_tensor(pred_map)

                sim_metric = cal_sim(pred_map.numpy(), gt_map)
                metric_dict["semantic"]['sim'].append(sim_metric)

                if n_viz["semantic"] < n_to_viz and n_query == 0:  # only save first query
                    # collect samples for viz
                    viz_dict["semantic"]['f0'].append(f0_im.cpu().squeeze())
                    viz_dict["semantic"]['f1'].append(f1_im.cpu().squeeze())
                    viz_dict["semantic"]['f0_co'].append([f0_co.cpu().squeeze().tolist()[n_query]])
                    viz_dict["semantic"]['name1'].append(name1)
                    viz_dict["semantic"]['name2'].append(name2)
                    viz_dict["semantic"]['gt_hmap_query'].append(gt_map_query)
                    viz_dict["semantic"]['gt_hmap'].append(gt_map)
                    viz_dict["semantic"]['pred_hmap'].append([pred_map])
                    viz_dict["semantic"]['sim'].append([sim_metric])
                    # predicted mask
                    if model.mask_loss:
                        pred_mask0 *= mask[i]
                        pred_mask1 *= mask[i+1]
                        viz_dict["semantic"]['pred_mask0'].append(pred_mask0.numpy())
                        viz_dict["semantic"]['pred_mask1'].append(pred_mask1.numpy())
                if n_viz["semantic"] < n_to_viz and n_query < 5 and n_query > 0: # we are gonna visualize first 5 points and heatmaps.
                    viz_dict["semantic"]['f0_co'][-1].append(f0_co.cpu().squeeze().tolist()[n_query])
                    viz_dict["semantic"]['pred_hmap'][-1].append(pred_map)
                    viz_dict["semantic"]['sim'][-1].append(sim_metric)
            n_viz["semantic"] += 1

        metric_dict["semantic"]['loss'].append(loss.item())
        metric_dict["semantic"]['pos_l'].append(pos_l.item())
        metric_dict["semantic"]['neg_l'].append(neg_l.item())
        if model.mask_loss:
            metric_dict["semantic"]['mask_loss'].append(mask_loss.item())
        else:
            metric_dict["semantic"]['mask_loss'].append(mask_loss)

        end = time.time()

        if itr % 10 == 0:
            print(f"SEMANTIC eval {split}: {itr:04d}/{loader_length:04d} loss:{loss.item():.2f} pos_l:{pos_l:.3f} neg_l:{neg_l:.3f} mask_loss:{mask_loss:.3f} time:{end-start:.2f}")
    
    if batch_spatial is not None:
        start = time.time()
        # unpack the batch
        image1 = batch_spatial['img1'].cuda()
        image2 = batch_spatial['img2'].cuda()
        co1 = batch_spatial['co1'].cuda()
        co2 = batch_spatial['co2'].cuda()
        mask2 = batch_spatial['binary_mask2']
        lang_feat = batch_spatial['lang_feat'].cuda()
        
        # get basic metrics
        logits, pos_l, neg_l = model.spatial_forward(image1, image2, co1, co2, lang_feat)
        loss = loss_fn(logits)
        
        # compute metrics and gather stuff for viz
        # for i in range(len(image1)):
        for i in range(2): # HACK only calculate normalized distance for first 2 pairs
            f0_im = image1[i].unsqueeze(0)
            f1_im = image2[i].unsqueeze(0)

            name1 = batch_spatial['name1'][i]
            name2 = batch_spatial['name2'][i]
            part = batch_spatial['part'][i]
            
            f0_lang_feat = lang_feat[i].unsqueeze(0)
            f1_lang_feat = lang_feat[i].unsqueeze(0)
            
            f0_co = co1[i][0].unsqueeze(0).unsqueeze(0)
            f1_co = co2[i][0].cpu()

            pred_map = model.get_heatmap(f0_im, f1_im, f0_co, f0_lang_feat, f1_lang_feat)
            pred_map = pred_map.cpu()
            # only on the object, fill rest with minimum value
            pred_map *= mask2[i]
            pred_map[pred_map < 0.0] = 0.0
            pred_map += torch.min(pred_map+((1-mask2[i])*100)) * (1 - mask2[i])
            pred_map = normalize_tensor(pred_map)

            # find argmax
            argmax_flattened = torch.argmax(pred_map)
            f1_co_pred = torch.unravel_index(argmax_flattened, pred_map.shape)
            f1_co_pred = torch.tensor([t.item() for t in f1_co_pred])
            # be careful if a coordinate is row or col
            f1_co_pred = f1_co_pred[[1,0]] # row col -> col row
            
            # normalize by image size
            pixel_distance = torch.sqrt(torch.sum(((f1_co / 224.0) - (f1_co_pred /224.0)) ** 2))
            metric_dict["spatial"]['dist'].append(pixel_distance)
            
            if n_viz["spatial"] < n_to_viz: 
                # collect samples for viz
                viz_dict["spatial"]['f0'].append(f0_im.cpu().squeeze())
                viz_dict["spatial"]['f1'].append(f1_im.cpu().squeeze())
                viz_dict["spatial"]['f0_co'].append(f0_co.cpu().squeeze().tolist())
                viz_dict["spatial"]['f1_co_pred'].append(f1_co_pred.cpu().tolist())
                viz_dict["spatial"]['f1_co'].append(f1_co.cpu().tolist())
                viz_dict["spatial"]['name1'].append(name1)
                viz_dict["spatial"]['name2'].append(name2)
                viz_dict["spatial"]['part'].append(part)
                viz_dict["spatial"]['pred_hmap'].append(pred_map)
                viz_dict["spatial"]['dist'].append(pixel_distance)
            n_viz["spatial"] += 1
        
        metric_dict["spatial"]['loss'].append(loss.item())
        metric_dict["spatial"]['pos_l'].append(pos_l.item())
        metric_dict["spatial"]['neg_l'].append(neg_l.item())

        end = time.time()

        if itr % 10 == 0:
            print(f"SPATIAL eval {split}: {itr:04d}/{loader_length:04d} loss:{loss.item():.2f} pos_l:{pos_l:.3f} neg_l:{neg_l:.3f} time:{end-start:.2f}")
    return n_viz, viz_dict, metric_dict

@torch.no_grad()
def eval_epoch(model, loader, loader_spatial, loss_fn, epoch, n_to_viz=100):

    model.eval()
    try:
        split = loader.dataset.split
    except:
        split = loader_spatial.dataset.split
    
    metric_dict = {"semantic": defaultdict(list), "spatial": defaultdict(list)}
    viz_dict = {"semantic": defaultdict(list), "spatial": defaultdict(list)}
    
    n_viz = {"semantic": 0, "spatial": 0}
    if loader is not None and loader_spatial is None:
        for itr, batch in enumerate(loader):
            batch_spatial = None
            n_viz, viz_dict, metric_dict = eval_step(model, itr, batch, batch_spatial, loss_fn, n_viz, n_to_viz, metric_dict, viz_dict, len(loader), split)
    
    if loader is None and loader_spatial is not None:
        for itr, batch_spatial in enumerate(loader_spatial):
            batch = None
            n_viz, viz_dict, metric_dict = eval_step(model, itr, batch, batch_spatial, loss_fn, n_viz, n_to_viz, metric_dict, viz_dict, len(loader_spatial), split)
    
    if loader is not None and loader_spatial is not None:
        for itr, (batch, batch_spatial) in enumerate(zip(loader, loader_spatial)):
            n_viz, viz_dict, metric_dict = eval_step(model, itr, batch, batch_spatial, loss_fn, n_viz, n_to_viz, metric_dict, viz_dict, len(loader), split)
    
    # log to wandb
    if wandb.run is not None:
        for m in ["semantic", "spatial"]:
            for name, metric_list in metric_dict[m].items():
                to_log = np.mean(metric_list)
                wandb.log({f"epoch/{name}_{m}_{split}":to_log, "epoch":epoch})
        
    return metric_dict, viz_dict


def build_dataset(data_args, aug_args):

    train_dataset_list = []
    train_dataset_eval_list = []
    val_dataset_list = []

    for name in data_args.datasets:
        assert name in ["objaverse"]
        if name == "objaverse":
            train_dataset = Objaverse_CC_cogvlm(split="train", load_in_ram=data_args.load_in_ram, mode="train", num_points=data_args.num_points, aug_params=aug_args)
            train_dataset_eval = train_dataset
            val_dataset = Objaverse_CC_cogvlm(split="test", load_in_ram=data_args.load_in_ram, mode="eval", num_points=data_args.num_points, test_selected=data_args.test_selected)

        train_dataset_list.append(train_dataset)
        train_dataset_eval_list.append(train_dataset_eval)
        val_dataset_list.append(val_dataset)

    train_dataset = CustomConcatDataset(train_dataset_list)
    train_dataset_eval = CustomConcatDataset(train_dataset_eval_list)
    val_dataset = CustomConcatDataset(val_dataset_list)

    train_sampler = BatchSampler(train_dataset.affordance_label_dict, data_args.train_samples, 2, data_args.pairs_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=data_args.num_workers)

    # HACK: reduce pairs_per_batch for inference, save some time
    train_sampler_eval = BatchSampler(train_dataset_eval.affordance_label_dict, data_args.eval_samples, 2, 10) # only select 10 pairs
    train_loader_eval = DataLoader(train_dataset_eval, batch_sampler=train_sampler_eval, num_workers=data_args.num_workers)

    val_sampler = BatchSampler(val_dataset.affordance_label_dict, data_args.eval_samples, 2, 10) # only select 10 pairs
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=data_args.num_workers)

    return train_loader, train_loader_eval, val_loader

def build_dataset_spatial(data_args, aug_args, spatial_partonly):

    train_dataset_list = []
    train_dataset_eval_list = []
    val_dataset_list = []

    for name in data_args.datasets:
        assert name in ["objaverse"]
        if name == "objaverse":
            train_dataset = Objaverse_CC_spatial(split="train", load_in_ram=data_args.load_in_ram, mode="train", num_points=data_args.num_points, aug_params=aug_args, spatial_partonly=spatial_partonly)
            train_dataset_eval = train_dataset
            val_dataset = Objaverse_CC_spatial(split="test", load_in_ram=data_args.load_in_ram, mode="eval", num_points=data_args.num_points, test_selected=data_args.test_selected, spatial_partonly=spatial_partonly)

        train_dataset_list.append(train_dataset)
        train_dataset_eval_list.append(train_dataset_eval)
        val_dataset_list.append(val_dataset)

    train_sampler = SpatialBatchSampler(train_dataset.num_pairs, data_args.train_samples, data_args.pairs_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=data_args.num_workers)

    # HACK: reduce pairs_per_batch for inference, save some time
    train_sampler_eval = SpatialBatchSampler(train_dataset_eval.num_pairs, data_args.eval_samples, 10) # only select 10 pairs
    train_loader_eval = DataLoader(train_dataset_eval, batch_sampler=train_sampler_eval, num_workers=data_args.num_workers)

    val_sampler = SpatialBatchSampler(val_dataset.num_pairs, data_args.eval_samples, 10) # only select 10 pairs
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=data_args.num_workers)

    return train_loader, train_loader_eval, val_loader


def setup_log_dir(exp_args):
    log_dir = os.path.join(
	"<#TODO: path to your dir>/experiment_logs",
	exp_args.exp_name,
	datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print('log dir:', log_dir)
    return log_dir, ckpt_dir

def main():

    # only global variable
    global GLOBAL_ITR
    GLOBAL_ITR = 0

    opt_cmd = options.parse_arguments(sys.argv[1:])
    cfg = options.set(opt_cmd=opt_cmd, verbose=True)

    # seeding
    torch.manual_seed(cfg.exp_args.seed)
    np.random.seed(cfg.exp_args.seed)
    random.seed(cfg.exp_args.seed)

    if not cfg.exp_args.debug:
        wandb_init(cfg.exp_args.exp_name)
        log_dir, ckpt_dir = setup_log_dir(cfg.exp_args)
    else:
        log_dir, ckpt_dir = None, None

    if cfg.training_args.semantic:
        train_loader, train_loader_eval, val_loader = build_dataset(cfg.data_args, cfg.aug_args) 
    else:
        train_loader, train_loader_eval, val_loader = None, None, None
    
    if cfg.training_args.spatial:
        train_loader_spatial, train_loader_eval_spatial, val_loader_spatial = build_dataset_spatial(cfg.data_args, cfg.aug_args, cfg.training_args.spatial_partonly) 
    else:
        train_loader_spatial, train_loader_eval_spatial, val_loader_spatial = None, None, None

    # setup model
    model_args = cfg.model_args
    model = ConditionedDino(model_args.use_last_layers, model_args.use_smaller_stride, model_args.use_featup, negneg_loss=cfg.training_args.negneg_loss, mask_loss=cfg.training_args.mask_loss, additional_model_args = cfg.additional_model_args).cuda()
    count_parameters(model)

    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg.optim_args.lr)
    loss_fn = NCESoftmaxLoss().cuda()
    
    for epoch in range(cfg.optim_args.epochs):
        print(f"Starting epoch {epoch:05d}...")
        train_epoch(model, optimizer, train_loader, train_loader_spatial, loss_fn, cfg.training_args.weight_spatial, cfg.training_args.mask_loss_weight)

        if cfg.training_args.semantic:
            train_loader_eval.batch_sampler.reset_rng() # so we see the same viz each time
            val_loader.batch_sampler.reset_rng()
        if cfg.training_args.spatial:
            train_loader_eval_spatial.batch_sampler.reset_rng() # so we see the same viz each time
            val_loader_spatial.batch_sampler.reset_rng()

        train_metric_dict, train_viz_dict = eval_epoch(model, train_loader_eval, train_loader_eval_spatial, loss_fn, epoch)
        val_metric_dict, val_viz_dict = eval_epoch(model, val_loader, val_loader_spatial, loss_fn, epoch)

        if log_dir is not None and epoch % 10 == 0:
            if cfg.training_args.semantic:
                tr_viz_out_dir = os.path.join(log_dir, "viz", f"e{epoch:04d}", "semantic_train")
                ts_viz_out_dir = os.path.join(log_dir, "viz", f"e{epoch:04d}", "semantic_test")
                os.makedirs(tr_viz_out_dir, exist_ok=True)
                os.makedirs(ts_viz_out_dir, exist_ok=True)
                visualize(train_viz_dict["semantic"], tr_viz_out_dir)
                visualize(val_viz_dict["semantic"], ts_viz_out_dir)
            if cfg.training_args.spatial:
                tr_viz_out_dir = os.path.join(log_dir, "viz", f"e{epoch:04d}", "spatial_train")
                ts_viz_out_dir = os.path.join(log_dir, "viz", f"e{epoch:04d}", "spatial_test")
                os.makedirs(tr_viz_out_dir, exist_ok=True)
                os.makedirs(ts_viz_out_dir, exist_ok=True)
                visualize_spatial(train_viz_dict["spatial"], tr_viz_out_dir)
                visualize_spatial(val_viz_dict["spatial"], ts_viz_out_dir)

            torch.save({
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                },
                os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            )

if __name__ == "__main__":
    main()
