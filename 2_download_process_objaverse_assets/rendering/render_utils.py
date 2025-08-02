import bpy # type: ignore
import os
import math
import random
import copy

import numpy as np

from mathutils import Vector, Matrix # type: ignore

def apply_all_transforms_to_object(obj):
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Make the object active and selected
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Apply all transforms: location, rotation, and scale
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Deselect the object and remove it as active
    obj.select_set(False)
    bpy.context.view_layer.objects.active = None
    
    print(f"All transforms applied to object '{obj.name}' and deselected.")

def enable_gpu_rendering():
    # Set the render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Access the Cycles preferences
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    
    # Refresh the device list
    cycles_prefs.refresh_devices()
    
    # Set the compute device type (CUDA, OPTIX, or OPENCL)
    # Change this to 'OPTIX' or 'OPENCL' as needed
    cycles_prefs.compute_device_type = 'CUDA'

    # Enable all available GPUs
    for device in cycles_prefs.devices:
        if device.type == 'CUDA':
            device.use = True
            print(f"Enabled {device.type} device: {device.name}")
        else:
            device.use = False

    # Optionally, set the tile size for optimal GPU performance
    bpy.context.scene.cycles.tile_size = 250

def get_3x4_RT_matrix_from_blender(cam):
    """
    Calculate the 3x4 rotation and translation matrix (RT matrix) from a Blender camera object.
    
    This function transforms the Blender camera's coordinate system into a format compatible
    with computer vision libraries by taking into account Blender's coordinate system and
    potential constraints on the camera's transformations.
    
    Parameters:
    cam (bpy.types.Object): The Blender camera object from which to extract the RT matrix.
    
    Returns:
    Matrix: A 3x4 Matrix that represents the rotation and translation of the camera in
            the computer vision coordinate system.
    """
    # Define the transformation matrix from Blender camera to computer vision camera coordinates
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1))
    )

    # Decompose the world matrix to get location and rotation
    location, rotation = cam.matrix_world.decompose()[0:2]

    # Compute the world to Blender camera rotation matrix by transposing the rotation matrix
    R_world2bcam = rotation.to_matrix().transposed()

    # Calculate the translation vector from world to Blender camera coordinates
    T_world2bcam = -1 * R_world2bcam @ location

    # Compute the rotation matrix from world to computer vision camera coordinates
    R_world2cv = R_bcam2cv @ R_world2bcam

    # Compute the translation vector from world to computer vision camera coordinates
    T_world2cv = R_bcam2cv @ T_world2bcam

    # Construct the 3x4 RT matrix by combining the rotation matrix and translation vector
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))

    return RT

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_calibration_matrix_K_from_blender(camera):
    """
    Compute the intrinsic calibration matrix (K) for a Blender perspective camera.

    Parameters:
    camera (bpy.types.Object): The Blender camera object.

    Returns:
    Matrix: A 3x3 matrix representing the camera's intrinsic parameters.
    
    Raises:
    ValueError: If the camera is not a perspective camera.
    """
    # Access the camera data
    camd = camera.data

    # Ensure the camera is a perspective camera
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')

    # Get the current scene
    scene = bpy.context.scene

    # Focal length in millimeters
    f_in_mm = camd.lens

    # Scale factor for resolution
    scale = scene.render.resolution_percentage / 100

    # Resolution in pixels
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y

    # Sensor size in millimeters
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)

    # Determine sensor fit based on aspect ratio and resolution
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )

    # Pixel aspect ratio
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    # Determine view factor based on sensor fit
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

    # Calculate pixel size in millimeters per pixel
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px

    # Scale factors for the intrinsic matrix
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Intrinsic matrix parameters
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # Assuming rectangular pixels (no skew)

    # Construct the intrinsic calibration matrix K
    K = Matrix(
        ((s_u, skew, u_0),
         (  0,  s_v, v_0),
         (  0,    0,   1))
    )

    return K

def clear_scene():
    # Remove all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Remove all cameras
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    # Remove all lights
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

def import_obj(obj_filepath):
    # Import the OBJ file without any rotation
    bpy.ops.wm.obj_import(filepath=obj_filepath, up_axis='Z')

    # After importing, the imported object should be the active object
    obj = bpy.context.active_object
    
    if obj is None:
        print(f"Failed to import the object from {obj_filepath}")
        return None
    
    print(f"Imported object '{obj.name}' from {obj_filepath}")
    return obj

def import_glb(glb_filepath):
    bpy.ops.import_scene.gltf(filepath=glb_filepath)

    # After importing, the imported object should be the active object
    obj = bpy.context.active_object
    
    if obj is None:
        print(f"Failed to import the object from {glb_filepath}")
        return None
    
    print(f"Imported object '{obj.name}' from {glb_filepath}")
    return obj


def setup_hdri_lighting(hdri_filepath):
    # Create a new world with HDRI lighting
    world = bpy.data.worlds.new("HDRI World")
    bpy.context.scene.world = world

    # Use nodes to set up HDRI lighting
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add nodes
    node_background = nodes.new(type='ShaderNodeBackground')
    node_environment = nodes.new(type='ShaderNodeTexEnvironment')
    node_output = nodes.new(type='ShaderNodeOutputWorld')

    # Set the HDRI file path
    node_environment.image = bpy.data.images.load(hdri_filepath)

    # Link nodes
    links.new(node_environment.outputs['Color'], node_background.inputs['Color'])
    links.new(node_background.outputs['Background'], node_output.inputs['Surface'])


def add_cameras_on_icosphere(density):
    # Create an icosphere
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=density, radius=2)
    icosphere = bpy.context.active_object

    # Store camera positions
    camera_positions = [v.co for v in icosphere.data.vertices]
    #camera_positions = [camera_positions[i] for i in [26,29,33,35]]
    
    # Delete the icosphere after getting positions
    bpy.ops.object.delete()

    # Create cameras at each vertex
    cameras = []
    for i, position in enumerate(camera_positions):
        camera_data = bpy.data.cameras.new(name=f'Camera_{i}')
        camera_obj = bpy.data.objects.new(name=f'Camera_{i}', object_data=camera_data)
        bpy.context.collection.objects.link(camera_obj)

        # Set camera location
        camera_obj.location = position

        # Look at the origin
        direction = Vector((0, 0, 0)) - position
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_obj.rotation_euler = rot_quat.to_euler()

        cameras.append(camera_obj)
    
    return cameras

def add_cameras_custom():
    camera_positions = np.load(os.path.join(os.getcwd(), "camera_positions.npy"))

    # Create cameras at each vertex
    cameras = []
    for i, position in enumerate(camera_positions):
        camera_data = bpy.data.cameras.new(name=f'Camera_{i}')
        camera_obj = bpy.data.objects.new(name=f'Camera_{i}', object_data=camera_data)
        bpy.context.collection.objects.link(camera_obj)

        # Set camera location
        camera_obj.location = position

        # Look at the origin
        direction = Vector((0, 0, 0)) - Vector(position)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_obj.rotation_euler = rot_quat.to_euler()

        cameras.append(camera_obj)
    
    return cameras


def add_random_cameras_on_sphere(density, elevation_min_deg, elevation_max_deg, radius=2):
    def random_point_on_sphere(radius, elevation_min_rad, elevation_max_rad):
        # Generate random elevation within the specified range
        elevation = random.uniform(elevation_min_rad, elevation_max_rad)
        
        # Generate random azimuth
        azimuth = random.uniform(0, 2 * math.pi)
        
        # Convert spherical coordinates to Cartesian coordinates
        x = radius * math.cos(azimuth) * math.cos(elevation)
        y = radius * math.sin(azimuth) * math.cos(elevation)
        z = radius * math.sin(elevation)
        
        return Vector((x, y, z))

    # Convert degrees to radians
    elevation_min_rad = math.radians(elevation_min_deg)
    elevation_max_rad = math.radians(elevation_max_deg)

    cameras = []
    for i in range(density):
        # Generate a random point on the sphere within the specified elevation range
        position = random_point_on_sphere(radius, elevation_min_rad, elevation_max_rad)

        # Create a new camera object
        camera_data = bpy.data.cameras.new(name=f'Camera_{i}')
        camera_obj = bpy.data.objects.new(name=f'Camera_{i}', object_data=camera_data)
        bpy.context.collection.objects.link(camera_obj)

        # Set camera location
        camera_obj.location = position

        # Look at the origin
        direction = Vector((0, 0, 0)) - position
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_obj.rotation_euler = rot_quat.to_euler()

        cameras.append(camera_obj)
    
    return cameras

def apply_global_rotation(obj, axis, angle_degrees):
    # Ensure object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Convert the angle from degrees to radians
    angle_radians = math.radians(angle_degrees)
    
    # Create a rotation matrix from the axis-angle
    rotation_matrix = Matrix.Rotation(angle_radians, 4, Vector(axis))
    
    # Apply the global rotation by multiplying the object's matrix_world
    obj.matrix_world = rotation_matrix @ obj.matrix_world
    
    # Update the object's data to ensure the transformation is applied
    bpy.context.view_layer.update()

def reset_rotation(obj):
    obj.matrix_world = Matrix(np.eye(4).tolist())
    bpy.context.view_layer.update()

def apply_random_xyz_rotation(obj):
    # Apply a random rotation around the X axis

    reset_rotation(obj)
    random_x_degrees = random.uniform(0, 25)
    apply_global_rotation(obj, axis=(1, 0, 0), angle_degrees=random_x_degrees)
    
    # Apply a random rotation around the Y axis
    random_y_degrees = random.uniform(0, 25)
    apply_global_rotation(obj, axis=(0, 1, 0), angle_degrees=random_y_degrees)
    
    # Apply a random rotation around the Z axis
    random_z_degrees = random.uniform(0, 25)
    apply_global_rotation(obj, axis=(0, 0, 1), angle_degrees=random_z_degrees)
    
    print(f"Applied random rotations: X={random_x_degrees}°, Y={random_y_degrees}°, Z={random_z_degrees}° to object '{obj.name}'")

def setup_compositor_nodes_label():
    """Set up the compositor nodes for depth output."""
    # Use the compositor
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    # Add Render Layers node
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (0, 0)
    # Add Composite node
    composite = tree.nodes.new(type='CompositorNodeComposite')
    composite.location = (400, 0)
    # Add File Output node for depth
    depth_output = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output.location = (400, -200)
    depth_output.label = 'Depth Output'
    depth_output.format.file_format = 'OPEN_EXR'
    
    # enable the Z buffer
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    # Link nodes
    links.new(render_layers.outputs['Image'], composite.inputs['Image'])
    links.new(render_layers.outputs['Depth'], depth_output.inputs[0])
    
    return depth_output

def setup_compositor_nodes_train():
    """Set up the compositor nodes for depth and alpha-based segmentation output."""

    # Enable transparency in the render settings
    bpy.context.scene.render.film_transparent = True
    
    # Enable the alpha pass in view layer settings
    bpy.context.scene.view_layers["ViewLayer"].use_pass_combined = True

    # Use the compositor
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Add Render Layers node
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (0, 0)

    # Add Composite node
    composite = tree.nodes.new(type='CompositorNodeComposite')
    composite.location = (400, 0)

    # Add File Output node for depth
    depth_output = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output.location = (400, -200)
    depth_output.label = 'Depth Output'
    depth_output.format.file_format = 'OPEN_EXR'

    # Add Math node to threshold the alpha (turn into binary)
    math_node = tree.nodes.new(type='CompositorNodeMath')
    math_node.operation = 'GREATER_THAN'
    math_node.inputs[1].default_value = 0.5  # Adjust threshold as needed
    math_node.location = (200, -400)

    # Add File Output node for binary segmentation
    segmentation_output = tree.nodes.new(type='CompositorNodeOutputFile')
    segmentation_output.location = (400, -400)
    segmentation_output.label = 'Binary Segmentation Output'
    segmentation_output.format.file_format = 'PNG'
    segmentation_output.format.color_mode = 'BW'  # Set to black and white

    # Enable the Z buffer and alpha pass
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_combined = True  # Includes alpha pass

    # Link nodes
    links.new(render_layers.outputs['Image'], composite.inputs['Image'])
    links.new(render_layers.outputs['Depth'], depth_output.inputs[0])
    links.new(render_layers.outputs['Alpha'], math_node.inputs[0])
    links.new(math_node.outputs[0], segmentation_output.inputs[0])

    return depth_output, segmentation_output

def render_images(output_path, obj, resolution_x, resolution_y, white_bg = True):
    """Render images and depth maps."""
    # Create output directories
    rgb_output_path = os.path.join(output_path, "rgb_images")
    depth_output_path = os.path.join(output_path, "depth_exr")
    segmentation_output_path = os.path.join(output_path, "segmentation")
    K_output_path = os.path.join(output_path, "K")
    RT_output_path = os.path.join(output_path, "RT")
    obj_pose_output_path = os.path.join(output_path, "obj_pose")
    
    os.makedirs(rgb_output_path, exist_ok=True)
    os.makedirs(depth_output_path, exist_ok=True)
    os.makedirs(K_output_path, exist_ok=True)
    os.makedirs(RT_output_path, exist_ok=True)
    os.makedirs(obj_pose_output_path, exist_ok=True)

    # Set render resolution
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.context.scene.cycles.samples = 100
    
    # use gpu
    bpy.context.scene.cycles.device = 'GPU'

    # Enable denoising in the render settings
    bpy.context.scene.cycles.use_denoising = True

    # Set denoiser type (e.g., "OPTIX" for NVIDIA GPUs, "OPENIMAGEDENOISE" for CPU)
    bpy.context.scene.cycles.denoiser = 'OPTIX'

    # Set up compositor nodes: white_bg means train mode.
    if white_bg:
        depth_output_node, segmentation_output_node = setup_compositor_nodes_train()
    else:
        depth_output_node = setup_compositor_nodes_label()

    # Render from each camera
    for cam_idx, camera in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):

        # apply_random_xyz_rotation(obj)

        bpy.context.scene.camera = camera

        # Set file paths
        rgb_filepath = os.path.join(rgb_output_path, f'{cam_idx:04d}.png')

        # Set RGB image output path
        bpy.context.scene.render.filepath = rgb_filepath

        # Set depth EXR output path
        depth_output_node.base_path = depth_output_path
        depth_output_node.file_slots[0].path = f'{cam_idx:04d}_'

        segmentation_output_node.base_path = segmentation_output_path
        segmentation_output_node.file_slots[0].path = f'{cam_idx:04d}_'
		
        # Render and write still
        bpy.ops.render.render(write_still=True)
  	    
        # getting the K and RT camera matrices
        K = get_calibration_matrix_K_from_blender(camera)
        RT = get_3x4_RT_matrix_from_blender(camera)
      
        np.save(os.path.join(K_output_path, f"{cam_idx:04d}.npy"), np.array(K))
        np.save(os.path.join(RT_output_path, f"{cam_idx:04d}.npy"), np.array(RT))

        pose_matrix = copy.deepcopy(np.array(obj.matrix_world))
        np.save(os.path.join(obj_pose_output_path, f"{cam_idx:04d}.npy"), pose_matrix)
