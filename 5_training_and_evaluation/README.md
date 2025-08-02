Change all the <#TODO: > to actual paths you're using.

0. `export PYTHONPATH=.`
1. Run `pip install git+https://github.com/openai/CLIP.git` and `cond_corr/gen_clip_test_features.py` to generate clip text features for all the functions.
2. Run `cond_corr/data/utils/make_dataset_config_objaverse.py` to make the dataset config for training contrastive functional part loss.
3. Run `cond_corr/data/utils/make_spatial_dataset_config_objaverse.py`to make the dataset config for training spatial constrastive loss.
4. Update the paths in `cond_corr/data/objaverse_loader_cogvlm.py` and `cond_corr/data/objaverse_loader_spatial.py`.
5. Run `python cond_corr/train.py --yaml=<path_to_config_file>` to start training. An example config file for our full model is included in `configs/Dino_MLP1024_objaverse_semantic+spatial_NegNegLoss_partOnly_bgAug.yaml`
6. After running, evaluate the models. Use `cond_corr/inference_2D_objaverse.py` and `cond_corr/inference_2D_HANDAL.py` to inference and evaluate label transfer metrics. Use `cond_corr/inference_2D_discovery_objaverse.py` and `cond_corr/inference_2D_discovery_HANDAL.py` to inference and evaluate correspondence discovery metrics.
7. You need to install SD-DINO from https://github.com/Junyi42/sd-dino if you want to run eval on SD and SD+DINO baselines. And you may need to run ManipVQA (https://github.com/SiyuanHuang95/ManipVQA) for ManipVQA baseline.
8. Finally, you can use `cond_corr/utils/gather_results.py` and `cond_corr/utils/gather_results_handal.py` to collect all results into a csv file, which will also include breakdowns by within-category and across-category.

