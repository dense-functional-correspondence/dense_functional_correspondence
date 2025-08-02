Change all the <#TODO: > to actual paths you're using.

0. `pip install objaverse` and install Blender 4.2.0 on your system.
1. run `download_objaverse_assets.py --matches_dir="../0_taxonomy_and_metadata/temp_test_only/verified_assets.json" --save_dir="<path_to_save>"` to download Objaverse assets.
2. run `geometry_preprocessing/wrap_clear_parents_and_normalize.py` to clean up the glbs and normalize them
4. run `geometry_preprocessing/extract_pointclouds.py` to extract a point cloud on the surface of the mesh
5. run `rendering/wrapper.py` to render the assets and you can change the camera positions in `render_glb.py`
6. run `postprocessing/convert_exr_depth.py` to convert .exr depth to .npy files
7. run `postprocessing/find_visible_points_objaverse.py` to generate a dictionary that maps the pixels on each image to the index of the object's pointcloud