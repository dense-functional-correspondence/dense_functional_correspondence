Within each file, there may be "<#TODO: path to ...>" placeholders that need to be filled in

1. Install olamma
2. Run `determine_objaverse_categories.py` to summarize Caption3D captions into a single noun.
3. Run `match_with_our_objects.py` to match with a user-defined list of object categories.
4. Run `verify_objaverse_categories.py` to verify the matches using Llama LLM.
5. Use `2_download_process_objaverse_assets` folder to download and render objaverse assets from the previous output `objaverse_llm_verified_categories.json`.
6. Subsample all the assets and manually filter the assets using `manual_verify_objaverse.py`, which provides a UI via matplotlib.
7. An example final result (containing object categories and relevant Objaverse assets) is `0_taxonomy_and_metadata/Objaverse/verified_assets.json`