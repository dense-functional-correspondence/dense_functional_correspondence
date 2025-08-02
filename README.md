# Dense Functional Correspondence

Official implementation for "Weakly-Supervised Learning of Dense Functional Correspondences".

## Installation

For preparing and processing data, run 
```
conda env create -f pseudo_label_environment.yml
conda activate pseudo_label
```

For training and evaluation, run 
```
conda env create -f func_corr_environment.yml
conda activate func_corr
```

## Running the Repo

Each folder contains its own README file for how to run the pipeline. Make sure to change any <#TODO: path to ...> placeholders to actual paths. And check if a file requires specifying arguments.

1. 0_taxonomy_and_metadata contains the final (object,function) pair taxonomy and example metadatas for datasets.

2. 1_objaverse_asset_selection contains the process for matching unlabelled objaverse assets to our taxonomy of objects. 

3. 2_download_process_objaverse_assets contains the process of downloading, processing, and rendering objaverse assets.

4. 3_pseudo_label contains the full pseudo-labeling process, which predicts functional part segments.

5. 4_eval_curation contains the process for selecting asset pairs, labelling functional alignment in 3D, and generating 2D ground-truth labels.

6. 5_training_and_evaluation contains the process for generating data configs, specifying training configs, training the model, and evaluating the model.
