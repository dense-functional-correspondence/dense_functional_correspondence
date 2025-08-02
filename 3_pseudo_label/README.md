Change all the <#TODO: > to actual paths you're using.

1. run `get_partnames_openai.py` to get part names and descriptions for each (object, action) pair. An example output is `0_taxonomy_and_metadata/Objaverse/obj2obj_fewshot_manually_processed.json`.
2. run `get_unique_parts.py` to combine duplicate part names and synthesize a unified description. An example output is `0_taxonomy_and_metadata/Objaverse/obj2part_descriptions.json`.
3. Install CogVLM and download their checkpoints. Refer to https://github.com/THUDM/CogVLM or follow:
```
conda env create -f pseudo_label_environment.yml
conda activate pseudo_label
# the next one is really important, otherwise xformers forces a new install of pytorch 2.2.0 with cuda 12.3 that breaks everything
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
python -m spacy download en_core_web_sm

git clone https://github.com/THUDM/CogVLM.git
cd CogVLM
export PYTHONPATH=.
```
You can use `python cli_demo_hf.py --from_pretrained THUDM/cogvlm-grounding-generalist-hf --quant 4` to check installation.
4. Symlink `cogvlm_scripts` to the same directory as CogVLM repo. Update checkpoint paths in `cogvlm_scripts/constants.py` (the downloaded models can be found in `~/.cache/huggingface/hub` by default).
5. Run `cogvlm_scripts/openai_api.py <port>` to start a CogVLM server. It will write the ip addresss and port to a txt file. If your GPU has less than 48GB RAM, run `cogvlm_scripts/openai_api_lowGPU.py <port>` to run on two smaller GPUs.
6. Run  `cogvlm_scripts/request_part.py` to run CogVLM to localize parts of objects.
7. By default, CogVLM will be queried 4 times to generate 4 bounding box predictions. So, use `aggregate_part_bbox.py` to use k-means to find the best bounding box.
8. [Optional] For small parts that require zooming in, e.g., specified by `0_taxonomy_and_metadata/Objaverse/crop_requirement.json`, run `crop_pad_images.py` to generate zoomin-in crops of the object parts using the bounding box from the previous step.
9. [Optional] Then, run `cogvlm_scripts/request_part_cropped.py` to run CogVLM to localize on the zoomed-in images.
10. [Optional] For the object parts that are edges (like edge of a knife), we further use SEDNet to get edge probabilities from the pointclouds. Refer to https://github.com/lly00412/SEDNet.git for installation. The edge requirement dictionary is in `0_taxonomy_and_metadata/Objaverse/edge_requirement.json`. Finally, run `python generate_edge_predictions.py configs/config_SEDNet_normal.yml Save multi_vote fold5drop` to produce the edge probabilities.
11. Finally, we can aggregate all the information onto the pointcloud using `heatmap_3dviz.py`.
12. Now, we can pseudo-label all the 2D images by running `pseudo_label.py`. For each (image, object part) pair, we now have a pseudo-labeled part mask.
13. Since training uses 224 by 224 images, run `resize_images.py` to resize everything to the desired size.
