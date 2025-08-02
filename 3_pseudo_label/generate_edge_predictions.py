import sys
import logging
import json
import os
import time
from read_config import Config
import argparse
config = Config(sys.argv[1])
GPU = config.gpu

os.environ['CUDA_VISIBLE_DEVICES'] = GPU

from shutil import copyfile
import numpy as np

from gen_test_vis import COLORS_TYPE, visual_labels
from src.dataset_segments import ori_simple_data
from src.smooth_normal_matrix import hpnet_process

def guard_mean_shift(ms, embedding, quantile, iterations, kernel_type="gaussian"):
    while True:
        _, center, bandwidth, cluster_ids = ms.mean_shift(
            embedding, 10000, quantile, iterations, kernel_type=kernel_type
        )
        if torch.unique(cluster_ids).shape[0] > 49:
            quantile *= 1.2
        else:
            break
    return center, bandwidth, cluster_ids

def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    EPS = np.finfo(np.float32).eps
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
                    [sin, cos, 0],
                    [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R    

def preprocess_data(points, normals):
    EPS = np.finfo(np.float32).eps
    means = np.mean(points, 0)
    points = (points - means)
    std = np.max(points, 0) - np.min(points, 0)
    points = points / (np.max(std) + EPS)

    S, U = np.linalg.eig(points.T @ points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points = (R @ points.T).T
    normals = (R @ normals.T).T
    return points, normals

if __name__ == "__main__":
    BASE_DIR = "<#TODO: path to the root rendered data directory>"
    asset_dir = "<#TODO: path to the pointcloud directory, output of 2_download_process_objaverse_assets/geometry_preprocessing/extract_pointclouds.py>"
    edge_requirement_dir = "<#TODO: path to edge_requirement.json>"
    edge_requirement = json.load(open(edge_requirement_dir))

    program_root = os.path.dirname(os.path.abspath(__file__)) + "/"
    sys.path.append(program_root + "src")

    import torch
    from src.SEDNet import SEDNet

    from src.segment_loss import EmbeddingLoss
    from src.segment_utils import SIOU_matched_segments_usecd, compute_type_miou_abc
    from src.segment_utils import to_one_hot, SIOU_matched_segments
    from src.mean_shift import MeanShift
    from src.segment_utils import SIOU_matched_segments

    # test configs
    HPNet_embed = True # ========================= default True 
    NORMAL_SMOOTH_W = 0.5  # =================== default 0.5
    Concat_TYPE_C6 = False # ====================== default False
    Concat_EDGE_C2 = False # ====================== default False
    INPUT_SIZE = 10000 # =====input pc num, default 10000
    my_knn = 64 # ==== default 64
    use_hpnet_type_iou = False
    drop_out_num = 2000 # ====== type seg rand drop  

    if HPNet_embed:
        print("uisng HPNet embeding way!!!!")

    SAVE_VIZ = not sys.argv[2] == "NoSave"

    # type 结果进行数据增强投票
    MULTI_VOTE = sys.argv[3] == "multi_vote"
    if MULTI_VOTE:
        print("type_multi_vote")

    # type 结果进行数据增强投票
    fold5Drop = sys.argv[4] == "fold5drop"
    if fold5Drop:
        print("type_fold5drop")  # ======= 效果好

    if_normals = config.normals
    userspace = ""
    Loss = EmbeddingLoss(margin=1.0)

    model = SEDNet(
            embedding=True,
            emb_size=128,
            primitives=True,
            num_primitives=6,
            loss_function=Loss.triplet_loss,
            mode=5,
            num_channels=6,
            combine_label_prim=True,   # early fusion
            edge_module=True,  # add edge cls module
            late_fusion=True,  
            nn_nb=my_knn  # default is 64
        )
    model_inst = SEDNet(
            embedding=True,
            emb_size=128,
            primitives=True,
            num_primitives=6,
            loss_function=Loss.triplet_loss,
            mode=5,
            num_channels=6,
            combine_label_prim=True,   # early fusion
            edge_module=True,  # add edge cls module
            late_fusion=True,    # ======================================
            nn_nb=my_knn  # default is 64
        )

    model = model.cuda( )
    model_inst = model_inst.cuda( )

    ms = MeanShift()

    model.eval()
    model_inst.eval()

    iterations = 50
    quantile = 0.015

    state_dict = torch.load(config.pretrain_model_path)
    state_dict = {k[k.find(".")+1:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
    model.load_state_dict(state_dict)

    state_dict = torch.load(config.pretrain_model_type_path)
    state_dict = {k[k.find(".")+1:]: state_dict[k] for k in state_dict.keys()} if list(state_dict.keys())[0].startswith("module.") else state_dict
    model_inst.load_state_dict(state_dict)

    save_gt = True

    categories = os.listdir(BASE_DIR)
    start_time = time.time()
    for cat_idx, category in enumerate(categories):
        print(f"\n{cat_idx}/{len(categories)}: Running SED-Net inference on {category} images!")
        base_category = category.split("---")[-1].replace(".glb","")
        uid = category.split("---")[0]
        req = edge_requirement[base_category]
        required = False
        # check if we need it
        for key, value in req.items():
            if value:
                required = True
        if not required:
            continue

        base_dir = os.path.join(BASE_DIR, category, "primitives")
        if os.path.exists(os.path.join(base_dir, "results", "edge.txt")): # already ran
            continue
        os.makedirs(base_dir, exist_ok = True)
        point_normals = np.load(os.path.join(asset_dir, f"{uid}.npy")) # 100000 * 6
        # sample 10k points
        sampled_point_normals = point_normals[np.random.choice(point_normals.shape[0], size=10000, replace=False)]

        points_ = sampled_point_normals[:, :3]
        normals_ = sampled_point_normals[:, 3:]
        np.save(os.path.join(base_dir, 'sampled_point_cloud.npy'), points_)
        np.save(os.path.join(base_dir, 'sampled_normals.npy'), normals_)
        points_, normals_ = preprocess_data(points_, normals_)
        points = torch.tensor(points_).unsqueeze(0).float().cuda()
        normals = torch.tensor(normals_).unsqueeze(0).float().cuda()

        with torch.no_grad():
            if if_normals:
                _input = torch.cat([points, normals], 2)
                primitives_log_prob = model(
                    _input.permute(0, 2, 1), None, False
                )[1]
                embedding, _, _, edges_pred = model_inst(
                    _input.permute(0, 2, 1), None, False
                )           
            else:
                primitives_log_prob = model(
                    points.permute(0, 2, 1), None, False
                )[1]
                embedding, _, _, edges_pred = model_inst(
                    points.permute(0, 2, 1), None, False
                )

            if MULTI_VOTE and not fold5Drop:
                points_big = points * 1.15
                if if_normals:
                    input = torch.cat([points_big, normals], 2)
                    embedding_big, primitives_log_prob_big = model(
                        input.permute(0, 2, 1), None, False
                    )[:2]
                else:
                    embedding_big, primitives_log_prob_big = model(
                        points_big.permute(0, 2, 1), None, False
                    )[:2]

                points_small = points * 0.85
                if if_normals:
                    input = torch.cat([points_small, normals], 2)
                    embedding_small, primitives_log_prob_small = model(
                        input.permute(0, 2, 1), None, False
                    )[:2]
                else:
                    embedding_small, primitives_log_prob_small = model(
                        points_small.permute(0, 2, 1), None, False
                    )[:2]              

                primitives_log_prob = (primitives_log_prob + primitives_log_prob_big + primitives_log_prob_small) / 3

            if fold5Drop and not MULTI_VOTE:
                # batch_points = None
                # batch_normals = None
                total_type_pred = torch.zeros_like(primitives_log_prob).flatten()
                primitives_log_prob_batch = None
                iter_times = 10000 // drop_out_num
                for i in range(iter_times):
                    index = torch.ones(points.shape, dtype=torch.bool).cuda()
                    index[:, i*drop_out_num:(i+1)*drop_out_num, :] = False
                    points_drop = points[index].reshape((1, 10000 - drop_out_num, 3))
                    normals_drop = normals[index].reshape((1, 10000 - drop_out_num, 3))
                    batch_points = points_drop
                    batch_normals = normals_drop
                    
                    if primitives_log_prob_batch is None:
                        if if_normals:
                            input = torch.cat([batch_points, batch_normals], 2)
                            primitives_log_prob_batch = model(
                                    input.permute(0, 2, 1), None, False
                                )[1]
                        else:
                            primitives_log_prob_batch = model(
                                    batch_points.permute(0, 2, 1), None, False
                                )[1]  
                    else:
                        if if_normals:
                            input = torch.cat([batch_points, batch_normals], 2)
                            primitives_log_prob_batch = torch.cat([primitives_log_prob_batch, model(
                                    input.permute(0, 2, 1), None, False
                                )[1]], dim=0)
                        else:
                            primitives_log_prob_batch = torch.cat([primitives_log_prob_batch, model(
                                    batch_points.permute(0, 2, 1), None, False
                                )[1]], dim=0)                     

                for i in range(iter_times):
                    index = torch.ones(primitives_log_prob.shape, dtype=torch.bool).cuda()
                    index[:, :, i*drop_out_num:(i+1)*drop_out_num] = False    
                    total_type_pred[index.flatten()] += primitives_log_prob_batch[i].flatten()

                primitives_log_prob += total_type_pred.reshape(primitives_log_prob.shape)

            if fold5Drop and MULTI_VOTE:
                """
                data augmentation

                """
                angles = [
                    torch.from_numpy(np.array(      
                            [[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]], dtype=np.float32)).cuda( ).unsqueeze(0), 
                    torch.from_numpy(np.array(      
                            [[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]], dtype=np.float32)).cuda( ).unsqueeze(0),                 
                ] 

                primitives_prob_total = None
                for R in angles:
                    normals_cur = torch.bmm(normals, R)
                    points_cur = torch.bmm(points, R)

                    if if_normals:
                        input = torch.cat([points_cur, normals_cur], 2)
                        primitives_log_prob_cur= model(input.permute(0, 2, 1), None, False)[1]
                    else:
                        primitives_log_prob_cur= model(points_cur.permute(0, 2, 1), None, False)[1]                

                    total_type_pred = torch.zeros_like(primitives_log_prob_cur).flatten()  # 6 x 1w
                    for i in range(5):
                        index = torch.ones(points.shape, dtype=torch.bool, device=torch.device("cuda"))
                        index[:, i*2000:(i+1)*2000, :] = False
                        points_drop = points_cur[index].reshape((1, 8000, 3))
                        normals_drop = normals_cur[index].reshape((1, 8000, 3))  # =========
                    
                        if if_normals:
                            _input = torch.cat([points_drop, normals_drop], 2)
                            primitives_log_prob_batch = model(
                                    _input.permute(0, 2, 1), None, False
                                )[1]
                        else:
                            primitives_log_prob_batch = model(
                                    batch_points.permute(0, 2, 1), None, False
                                )[1] 
                        index = torch.ones(primitives_log_prob_cur.shape, dtype=torch.bool, device=torch.device("cuda"))
                        index[:, :, i*2000:(i+1)*2000] = False    
                        total_type_pred[index.flatten() ] += primitives_log_prob_batch.flatten()

                    primitives_log_prob_cur += total_type_pred.reshape(primitives_log_prob.shape)     

                    if primitives_prob_total is None:
                        primitives_prob_total =  primitives_log_prob_cur
                    else:
                        primitives_prob_total += primitives_log_prob_cur

                primitives_log_prob = primitives_prob_total     

        pred_primitives = torch.max(primitives_log_prob[0], 0)[1].data.cpu().numpy()

        primitives_prob_total = None
        index = None
        total_type_pred = None

        if HPNet_embed:
            embedding = hpnet_process(embedding.transpose(1, 2), points, normals, id=None, 
                types=primitives_log_prob.transpose(1, 2) if Concat_TYPE_C6 else None,
                edges=edges_pred.transpose(1, 2) if Concat_EDGE_C2 else None,
                normal_smooth_w=NORMAL_SMOOTH_W, CHUNK=1000
            )
            embedding = torch.nn.functional.normalize(embedding[0], p=2, dim=1)

        else:
            embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)

        _, _, cluster_ids = guard_mean_shift(
                ms, embedding, quantile, iterations, kernel_type="gaussian"
            )
        weights = to_one_hot(cluster_ids, np.unique(cluster_ids.data.data.cpu().numpy()).shape[
            0])
        cluster_ids = cluster_ids.data.cpu().numpy()

        # HACK: modified from https://github.com/yuanqili78/SED-Net/blob/430b36210818d20c7d8fc4c444c86ebe75a1c83e/src/eval_utils.py#L183
        Mapping = { "plane": 1,
                    "cone": 3,
                    "cylinder": 4,
                    "sphere": 5,
                    "open-spline": 2,
                    "closed-spline": 0}
        # red is closed_spline, green is plane, blue is open-spline, cyan is cone, magenta is cylinder, Yellow is sphere
        COLORS = np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255],[0, 255, 255],[255, 0, 255],[255, 255, 0]])
        if SAVE_VIZ:
            pred_primitives[pred_primitives == 0] = 0
            pred_primitives[pred_primitives == 6] = 0
            pred_primitives[pred_primitives == 7] = 0
            pred_primitives[pred_primitives == 9] = 0
            pred_primitives[pred_primitives == 8] = 2

            os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
            np.savetxt(os.path.join(base_dir, "results", "inst.txt"), cluster_ids, fmt="%d")
            np.savetxt(os.path.join(base_dir, "results", "type.txt"), pred_primitives, fmt="%d")

            type_vis = visual_labels(points_, pred_primitives.astype(np.compat.long), COLORS)
            inst_vis = visual_labels(points_, cluster_ids.astype(np.compat.long), COLORS_TYPE)
            np.savetxt(os.path.join(base_dir, "results", "Vis_inst.txt"), inst_vis, fmt="%0.4f", delimiter=";")
            np.savetxt(os.path.join(base_dir, "results", "Vis_type.txt"), type_vis, fmt="%0.4f", delimiter=";")

            if edges_pred is not None:
                edges_pred = torch.softmax(edges_pred, dim=1).transpose(1, 2).squeeze(0).cpu().numpy()  # [N, 2]
                np.savetxt(os.path.join(base_dir, "results", "edge.txt"), edges_pred, fmt="%0.4f", delimiter=";")

        print(f"Running time: {time.time() - start_time} seconds")