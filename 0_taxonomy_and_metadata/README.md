1. `action2objects.json` -- the final taxonomy of functions and associated objects that can perform the function. `Objaverse/objaverse_action2objects.json` and `Objaverse/objaverse_object2actions.json` are subsets specific to Objaverse assets. `HANDAL/HANDAL_action2objects.json` and `HANDAL/HANDAL_object2actions.json` are subsets specific to HANDAL assets.
2. `Objaverse/verified_assets.json` contains the final list of asset uuids for each relevant object category.
3. `Objaverse/obj2part2function.json` contains the dictionary that maps objects to their functional parts, and functional parts to the function.
4. `Objaverse/obj2part_descriptions.json` contains a dictionary that maps objects to their functions parts, and each functional part has a description.
5. `Objaverse/obj2obj_fewshot_manually_processed.json` contains the manually processed dictionary that maps object to function to functional part descriptions.
6. `Objaverse/duplicated_parts.json` indicates parts that are duplicated.
7. `Objaverse/crop_requirement.json` and `edge_requirement.json` detail which functional part needs zooming in and predicting edge probabilities.
8. `Objaverse/xxx_split.json` shows the train / test split of the assets.
9. Similar structure for the HANDAL dataset.
10. `hdris` folder contains an example hdri for rendering