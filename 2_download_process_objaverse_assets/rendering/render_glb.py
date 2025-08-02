import os
import sys
import numpy as np
import copy
# Get the directory of the current file -- needed because of blender
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Append this directory to sys.path
if current_file_directory not in sys.path:
    sys.path.append(current_file_directory)

import render_utils as render_utils

def main(glb_filepath, hdri_filepath, output_path, density, resolution_x, resolution_y):
    """Main function to set up the scene and render."""
    # Clear the scene
    render_utils.clear_scene()

    # Enable GPU rendering
    render_utils.enable_gpu_rendering()

    # Import GLB file
    obj = render_utils.import_glb(glb_filepath)

    # OPTIONAL: apply random rotations, for labelling
    render_utils.apply_random_xyz_rotation(obj)

    # Copy pose matrix
    pose_matrix = copy.deepcopy(np.array(obj.matrix_world))

    # Set up HDRI lighting
    render_utils.setup_hdri_lighting(hdri_filepath)

    # Add cameras around the object, either icosphere, or custom sets of cameras
    # render_utils.add_random_cameras_on_sphere(30, -60, 60)
    # render_utils.add_cameras_on_icosphere(3)
    render_utils.add_cameras_custom()

    # Render images and depth maps
    render_utils.render_images(output_path, obj, resolution_x, resolution_y, white_bg=True)

    np.save(os.path.join(output_path, "object_pose.npy"), pose_matrix)

if __name__ == "__main__":
    import sys

    # Get arguments from the command line
    glb_filepath = sys.argv[-6]
    hdri_filepath = sys.argv[-5]
    output_path = sys.argv[-4]
    density = int(sys.argv[-3])
    resolution_x = int(sys.argv[-2])
    resolution_y = int(sys.argv[-1])

    # Run the main function
    main(glb_filepath, hdri_filepath, output_path, density, resolution_x, resolution_y)
