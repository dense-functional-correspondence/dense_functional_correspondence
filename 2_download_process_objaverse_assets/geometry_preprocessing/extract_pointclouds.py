import os
import trimesh
import numpy as np
import glob
from joblib import Parallel, delayed

def get_point_cloud(mesh):
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, 100000)
    normals = mesh.face_normals[face_indices]

    sampled_points = np.concatenate([sampled_points, normals], axis=-1)

    return sampled_points

def join_scene(scene):
    geometries = list(scene.geometry.values())
    combined_mesh = trimesh.util.concatenate(geometries)

    return combined_mesh

def merge_and_normalize_obj_files(item, output_directory):

    try:
        point_cloud_file = os.path.join(output_directory, item.split('/')[-1].replace('.glb','.npy'))
        scene = trimesh.load(item)
        mesh = join_scene(scene)

        # Sample 100,000 points from the mesh surface
        sampled_points, face_indices = trimesh.sample.sample_surface(mesh, 100000)
        normals = mesh.face_normals[face_indices]
        
        sampled_points = np.concatenate([sampled_points, normals], axis=-1)
        # Save the sampled point cloud to an NPY file
        np.save(point_cloud_file, np.array(sampled_points))
    except Exception as e:
        print(e)

def main():
    data_path  = "<#TODO: path to the normalized glbs>"
    output_path = "<#TODO: path to the output point clouds>"

    os.makedirs(output_path, exist_ok=True)

    items = sorted(glob.glob(f"{data_path}/*"))

    Parallel(n_jobs=8, backend='multiprocessing', verbose=10)(
        delayed(merge_and_normalize_obj_files)(item, output_path) for item in items)

if __name__ == "__main__":
    main()
