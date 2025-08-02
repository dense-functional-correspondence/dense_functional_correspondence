Change all the <#TODO: > to actual paths you're using.

1. Put all the seen test set renders and pointclouds under `objaverse_test/seen` folder. Run `make_meshes_list_objaverse.py` locally to manually pick the asset pairs you want to label for each function.
2. Run `transform_annotator_objaverse.py` in Blender to annotate the functionally equivalent 3D alignment of two assets. You can import the pair, label the 3D bbox for the functional part, align the functional parts, and export the annotations.
3. Then, run `3d_to_2d.py` to convert the 3D annotation to 2D dense correspondences. By default, this will output dense correspondences for 6 different view pairs.
4. Run `select_annotations.py` to manually select the view pairs that work well, i.e., check that the annotated dense correspondences are reasonable.
5. Finally, run `resize_annotations.py` to resize the dense correspondence annotation to 224 by 224 resolution. The result will be the final ground-truth annotations we use to evaluate.

We will provide download links to our human-labeled annotations for both Objaverse and HANDAL.