import bpy # type: ignore
import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to save the object file",
)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_non_meshes():
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def clear_parents():
    bpy.context.view_layer.objects.active = None
    for obj in scene_root_objects():
        
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

def clear_nonmesh():
    for obj in scene_non_meshes():
        bpy.data.objects.remove(obj)

def join_mesh():
    bpy.context.view_layer.objects.active = None
    objects = []
    for obj in scene_meshes():
        obj.select_set(True)
        objects.append(obj)

    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()

def normalize_object(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    vertices = np.array([v.co for v in obj.data.vertices])
    obj.scale = obj.scale * 0.5 / np.max(np.abs(vertices))
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

def load_object(path):
    bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)

for obj in bpy.data.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

name = os.path.split(args.object_path)[1].replace('.glb', '')
load_object(args.object_path)
clear_parents()
clear_nonmesh()
join_mesh()

assert len(bpy.data.objects) == 1

obj = bpy.data.objects[0]

normalize_object(obj)
obj.name = name

out_dir = os.path.split(args.output_path)[0]
os.makedirs(out_dir, exist_ok=True)

bpy.ops.export_scene.gltf(filepath=args.output_path)