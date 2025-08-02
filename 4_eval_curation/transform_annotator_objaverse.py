import bpy
import os
import json
import numpy as np
import mathutils


ACTION = "cut-with"  # you need to specify the action
mode = "seen"
base_dir = "<#TODO: path to the base directory for annotation>"
MESHES_LIST_FILE = os.path.join(base_dir, f"lists_per_action/{mode}/{ACTION}.txt")
MESHES_PATH = os.path.join(base_dir, f"objaverse_test/{mode}")
EXPORT_PATH = os.path.join(base_dir, f"labeled_transforms/{mode}/{ACTION}")

# Store the current index in Blender's properties to persist between sessions
def get_current_index(context):
    return context.scene.get("mesh_import_index", 0)

def set_current_index(context, index):
    context.scene["mesh_import_index"] = index

class ImportMeshesOperator(bpy.types.Operator):
    """Import two meshes from a predefined list"""
    bl_idname = "object.import_meshes"
    bl_label = "Import Meshes"
    
    def execute(self, context):
        # Load the list of meshes from the file
        with open(MESHES_LIST_FILE, 'r') as file:
            meshes = file.read().splitlines()
        
        # Get the current index
        current_index = get_current_index(context)
        
        # Ensure there are enough meshes left to import
        if current_index + 1 > len(meshes):
            self.report({'ERROR'}, "Not enough meshes left to import")
            return {'CANCELLED'}
        
        line = meshes[current_index]
        mesh0, mesh1 = line.split(", ")

        mesh0_uid = mesh0.split("---")[0]
        mesh0_p = f"{MESHES_PATH}/{mesh0}/{mesh0_uid}.glb"
        mesh1_uid = mesh1.split("---")[0]
        mesh1_p = f"{MESHES_PATH}/{mesh1}/{mesh1_uid}.glb"

        bpy.ops.import_scene.gltf(filepath=mesh0_p)
        obj = bpy.context.active_object
        obj.name = mesh0

        bpy.ops.import_scene.gltf(filepath=mesh1_p)
        obj = bpy.context.active_object
        obj.name = mesh1

        # Update the index
        set_current_index(context, current_index + 1)
        
        self.report({'INFO'}, f"Source mesh {mesh0} and target mesh {mesh1} imported")
        return {'FINISHED'}

class ImportMeshesOperator(bpy.types.Operator):
    """Import two meshes from a predefined list"""
    bl_idname = "object.import_meshes"
    bl_label = "Import Meshes"
    
    def execute(self, context):
        # Load the list of meshes from the file
        with open(MESHES_LIST_FILE, 'r') as file:
            meshes = file.read().splitlines()
        
        # Get the current index
        current_index = get_current_index(context)
        
        # Ensure there are enough meshes left to import
        if current_index + 1 > len(meshes):
            self.report({'ERROR'}, "Not enough meshes left to import")
            return {'CANCELLED'}
        
        line = meshes[current_index]
        mesh0, mesh1 = line.split(", ")

        mesh0_uid = mesh0.split("---")[0]
        mesh0_p = f"{MESHES_PATH}/{mesh0}/{mesh0_uid}.glb"
        mesh1_uid = mesh1.split("---")[0]
        mesh1_p = f"{MESHES_PATH}/{mesh1}/{mesh1_uid}.glb"

        # Import mesh0
        bpy.ops.import_scene.gltf(filepath=mesh0_p)
        obj0 = bpy.context.active_object
        obj0.name = mesh0
        
        # Create a cube for mesh0 and resize to the bounding box
        bbox_min = mathutils.Vector(obj0.bound_box[0])
        bbox_max = mathutils.Vector(obj0.bound_box[6])
        bbox_size = bbox_max - bbox_min
        bbox_center = (bbox_max + bbox_min) / 2
        
        bpy.ops.mesh.primitive_cube_add(location=bbox_center)
        bbox_cube0 = bpy.context.active_object
        bbox_cube0.scale = bbox_size / 2  # Scale the cube to half the bbox size
        bbox_cube0.scale *= 1.05
        bbox_cube0.name = f"{mesh0}_bbox"
        bbox_cube0.color = (1,0,0,0.3)

        # Import mesh1
        bpy.ops.import_scene.gltf(filepath=mesh1_p)
        obj1 = bpy.context.active_object
        obj1.name = mesh1
        
        # Create a cube for mesh1 and resize to the bounding box
        bbox_min = mathutils.Vector(obj1.bound_box[0])
        bbox_max = mathutils.Vector(obj1.bound_box[6])
        bbox_size = bbox_max - bbox_min
        bbox_center = (bbox_max + bbox_min) / 2
        
        bpy.ops.mesh.primitive_cube_add(location=bbox_center)
        bbox_cube1 = bpy.context.active_object
        bbox_cube1.scale = bbox_size / 2  # Scale the cube to half the bbox size
        bbox_cube1.scale *= 1.05
        bbox_cube1.name = f"{mesh1}_bbox"
        bbox_cube1.color = (0,0,1,0.3)
        
        # Update the index
        set_current_index(context, current_index + 1)
        
        self.report({'INFO'}, f"Source mesh {mesh0} and target mesh {mesh1} imported with bounding boxes")
        return {'FINISHED'}

class ResetImportIndexOperator(bpy.types.Operator):
    """Reset the mesh import index"""
    bl_idname = "object.reset_import_index"
    bl_label = "Reset Import Index"
    
    def execute(self, context):
        # Reset the index to start from the beginning of the list
        set_current_index(context, 0)
        self.report({'INFO'}, "Mesh import index reset")
        return {'FINISHED'}
    
class ExportTransformsOperator(bpy.types.Operator):
    """Export the relative transforms of the two selected objects to JSON"""
    bl_idname = "object.export_transforms"
    bl_label = "Export Transforms"
    
    def execute(self, context):
        selected_objects = [x for x in bpy.data.objects if "_bbox" not in x.name]
        if len(selected_objects) != 2:
            self.report({'ERROR'}, "Select exactly two objects")
            return {'CANCELLED'}
        
        obj1, obj2 = selected_objects[0], selected_objects[1]
        matrix1 = np.array(obj1.matrix_world)
        matrix2 = np.array(obj2.matrix_world)
        
        output_directory = os.path.join(EXPORT_PATH, f"{obj1.name}|||{obj2.name}")
        
        if not os.path.exists(output_directory):
            output_count = 0
            os.makedirs(output_directory)
        else:
            output_count = len(os.listdir(output_directory))

        subdirectory = os.path.join(output_directory, f"{output_count:04d}")
        os.makedirs(subdirectory, exist_ok=True)

        np.save(os.path.join(subdirectory, f"{obj1.name}.npy"), matrix1) 
        np.save(os.path.join(subdirectory, f"{obj2.name}.npy"), matrix2) 

        obj1_bbox_name = f"{obj1.name}_bbox"
        obj2_bbox_name = f"{obj2.name}_bbox"

        bbox1 = bpy.data.objects[obj1_bbox_name]
        bbox2 = bpy.data.objects[obj2_bbox_name]

        apply_scale_to_object(bbox1)
        apply_scale_to_object(bbox2)

        set_origin_to_bounding_box(bbox1)
        set_origin_to_bounding_box(bbox2)

        bbox_matrix1 = np.array(bbox1.matrix_world)
        bbox_matrix2 = np.array(bbox2.matrix_world)
        
        np.save(os.path.join(subdirectory, f"{obj1.name}_bbox.npy"), bbox_matrix1) 
        np.save(os.path.join(subdirectory, f"{obj2.name}_bbox.npy"), bbox_matrix2) 

        export_object_to_obj(bbox1, os.path.join(subdirectory, f"{obj1_bbox_name}.obj"))
        export_object_to_obj(bbox2, os.path.join(subdirectory, f"{obj2_bbox_name}.obj"))
        
        self.report({'INFO'}, f"Transforms exported to npy in {subdirectory}")
        return {'FINISHED'}

class SimpleTransformExporterPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Simple Transform Exporter"
    bl_idname = "OBJECT_PT_simple_transform_exporter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Transform Exporter"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("object.import_meshes", text="Import Meshes")
        row = layout.row()
        row.operator("object.export_transforms", text="Export Transforms")
        row = layout.row()
        row.operator("object.reset_import_index", text="Reset Import Index")


def export_object_to_obj(obj, file_path):
    """
    Export the specified object to an OBJ file.
    
    Parameters:
    object_name (str): The name of the object to export.
    file_path (str): The file path (including file name) where the OBJ file will be saved.
    
    Returns:
    bool: True if the export was successful, False if the object was not found.
    """
    # Get the object by name
    obj.location=(0,0,0)
    obj.rotation_euler=(0,0,0)
    
    if obj:
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Select the object to export
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Export the selected object to OBJ
        #bpy.ops.export_scene.obj(filepath=file_path, use_selection=True)
        bpy.ops.wm.obj_export(filepath=file_path, export_selected_objects=True)

        print(f"Exported '{obj.name}' to '{file_path}'.")
        return True
    else:
        print(f"Object '{obj.name}' not found.")
        return False

def set_origin_to_bounding_box(obj):
    """
    Set the object's origin to the center of its bounding box.
    
    Parameters:
    object_name (str): The name of the object to adjust.
    
    Returns:
    bool: True if the operation was successful, False if the object was not found.
    """
    
    if obj:
        # Ensure we are in object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Select and activate the object
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Set the origin to the bounding box center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        print(f"Set origin to bounding box center for object '{obj.name}'.")
        return True
    else:
        print(f"Object '{obj.name}' not found.")
        return False

def apply_scale_to_object(obj):
    """
    Apply scale transform to the specified object and reset its scale to (1, 1, 1).
    
    Parameters:
    object_name (str): The name of the object to apply the scale to.
    
    Returns:
    bool: True if the scale was successfully applied, False if the object was not found.
    """
    if obj:
        # Ensure we are in object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Set the object as active
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Apply the scale transformation
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Deselect the object after the operation
        obj.select_set(False)

        # Confirm the scale has been reset
        print(f"Applied scale for object '{obj.name}'. New scale: {obj.scale}")
        return True
    else:
        print(f"Object '{obj.name}' not found.")
        return False

def register():
    bpy.utils.register_class(ImportMeshesOperator)
    bpy.utils.register_class(ExportTransformsOperator)
    bpy.utils.register_class(ResetImportIndexOperator)
    bpy.utils.register_class(SimpleTransformExporterPanel)

def unregister():
    bpy.utils.unregister_class(ImportMeshesOperator)
    bpy.utils.unregister_class(ExportTransformsOperator)
    bpy.utils.unregister_class(ResetImportIndexOperator)
    bpy.utils.unregister_class(SimpleTransformExporterPanel)

if __name__ == "__main__":
    register()
