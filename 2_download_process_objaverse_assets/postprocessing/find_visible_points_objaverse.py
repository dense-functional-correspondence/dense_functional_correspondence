import trimesh
import os
import cv2
import numpy as np
import copy
import math
import heapq
import random
import json

from scipy.spatial import KDTree, cKDTree

from joblib import Parallel, delayed

                               #obj pts       bp_pts
def find_points_within_epsilon(source_points, query_points, already_visible, epsilon):
    """
    Find all query points within an epsilon distance from any source point.

    :param source_points: np.ndarray, shape (n_source, 3) - Source point cloud
    :param query_points: np.ndarray, shape (n_query, 3) - Query point cloud
    :param already_visible: set, Points that were already found to be visible before
    :param epsilon: float - Maximum distance to consider points as neighbors
    :return: np.ndarray - Indices of query points within epsilon distance from any source point
    """
    # Build a KDTree from the source points
    tree = KDTree(source_points)
    
    indices_keep = set()
    indices_keep_list = []
    query_indices = []
    distances_list, indices_list = tree.query(query_points, distance_upper_bound=epsilon, k=10)
    
    for i, (distances, indices) in enumerate(zip(distances_list, indices_list)):
        indices = [ind for dist, ind in zip(distances, indices) if not math.isinf(dist)]
        
        # nothing was visible, skip
        if len(indices) == 0:
            continue
        
        # the first one we'll just add the nearest one
        if len(indices_keep) == 0:
            indices_keep.add(indices[0])
            indices_keep_list.append(indices[0])
            query_indices.append(i)
        else:
            # only pick points that weren't choesn before in this loop
            valid_indices = [x for x in indices if x not in indices_keep]
            if len(valid_indices) == 0:
                continue

            if len(already_visible) > 0:
                # prioritize points that were seen in previously processed images
                new_valid_indices = [i for i in valid_indices if i in already_visible]
                if len(new_valid_indices) > 0:
                    valid_indices = new_valid_indices
            
            # take the nearest one
            indices_keep.add(valid_indices[0])
            indices_keep_list.append(valid_indices[0])
            query_indices.append(i)
    
    assert len(indices_keep) == len(indices_keep_list)
    assert len(indices_keep_list) ==  len(query_indices)

    return np.array(indices_keep_list), np.array(query_indices)

def get_pixel_grid(H, W):
    ### creating pixel points
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.array([x.flatten(), y.flatten()]).T

    x_co = coords[:, 0]
    y_co = coords[:, 1]
    z_co = np.ones_like(x_co)

    im_pixel_pts = np.stack([x_co, y_co, z_co]).T
    return im_pixel_pts

def backproject_depth(depth, K, R, T):
    K_inv = np.linalg.inv(K)
    H, W = depth.shape

    # list of pixels, (H*W x (X, Y, 1)
    im_pts = get_pixel_grid(H, W)

    ray_idx = depth != 0.0

    ray_idx = ray_idx.flatten()
    ray_idx = np.where(ray_idx)[0]
    c_depths = depth.flatten()[ray_idx]
    
    im_pts = im_pts[ray_idx]
    # unit C1 ray directions in C1 coordinate-system
    c_ray_dirs = K_inv @ im_pts.T
    c_pts = c_ray_dirs * c_depths

    bckpj_depth = c_pts.T
    
    return bckpj_depth, im_pts[:,:2] # don't need the 3rd dim anymore

def load_pc(path):
    obj_pts = np.load(path)[:,:3]
    init_R = trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0])[:3,:3]
    obj_pts = (init_R @ obj_pts.T).T
    return obj_pts

def find_visible_points(obj_pts, already_visible, depth, K, RT):
    depth[depth==depth.max()] = 0

    R = RT[:3,:3]
    T = RT[:,3]
    bp_pc, im_pts = backproject_depth(depth, K, R, T)
    obj_pts = (R @ obj_pts.T).T + T # converting obj pts to camera coordinates
    
    pt_idxs, query_idxs = find_points_within_epsilon(obj_pts, bp_pc, already_visible, 0.01)
    obj_pts = obj_pts[pt_idxs]
    im_pts = im_pts[query_idxs]

    # debugging viz
    # bp_pc = trimesh.PointCloud(bp_pc, colors=(0,0,255))
    # obj_pc = trimesh.PointCloud(obj_pts, colors=(255,0,0))
    # scene = trimesh.Scene([bp_pc, obj_pc])
    # scene.show()

    already_visible.update(set(pt_idxs))

    return pt_idxs, im_pts

def process_pts(render_dir, pc_dir):
    obj_name = render_dir.split('/')[-1].split("---")[0]
    out_dct = {}
    obj_pts = load_pc(os.path.join(pc_dir, f"{obj_name}.npy"))

    images = np.arange(0, len(os.listdir(os.path.join(render_dir, "rgb_images"))))
    already_visible = set()

    for im_idx in images:

        pose = np.load(os.path.join(render_dir, "obj_pose", f"{im_idx:04d}.npy"))
        
        pc = trimesh.PointCloud(copy.deepcopy(obj_pts))
        pc.apply_transform(pose)
        pc = np.array(pc.vertices)

        K = np.load(os.path.join(render_dir, "K", f"{im_idx:04d}.npy"))
        RT = np.load(os.path.join(render_dir, "RT", f"{im_idx:04d}.npy"))
        depth = np.load(os.path.join(render_dir, "depth_npy", f"{im_idx:04d}.npy")) 

        vis_idxs, proj_vis_pts = find_visible_points(pc, already_visible, depth, K, RT)

        # debugging
        # im = np.zeros((490,490))
        # im[proj_vis_pts[:,1], proj_vis_pts[:,0]] = 255
        # cv2.imwrite(f"test_viz_dir/{im_idx:04d}.png", im)

        out_dct[str(im_idx)] = {
            "vis_pt_idx": vis_idxs,
            "proj_2D_loc": proj_vis_pts[:,[1,0]]
        }

    output_path = os.path.join(render_dir, "visible_pt_dict.npz")
    np.savez(output_path, **out_dct)

def main():

    render_root = "<#TODO: path to rendered images>"
    pc_root = "<#TODO: path to extracted pointclouds>"
    
    items = os.listdir(render_root)
    
    render_dirs = [os.path.join(render_root, item) for item in items]

    Parallel(n_jobs=8, backend='sequential', verbose=10)(
        delayed(process_pts)(render_dir, pc_root) for render_dir in render_dirs)
    
if __name__ == "__main__":
    main()
