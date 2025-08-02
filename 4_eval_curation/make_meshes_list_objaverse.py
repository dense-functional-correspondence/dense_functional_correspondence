import json
import os
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

def make_grid_image(img_dir):
    image_paths = os.listdir(img_dir)
    image_paths = [x for x in image_paths if x[0] != "."] # avoid .DS_Store
    image_paths.sort()
    image_paths = image_paths[:9]
    # Load the images from file paths
    images = [mpimg.imread(os.path.join(img_dir, img_path)) for img_path in image_paths]

    # Step 2: Preprocess the images into a single grid (3x3)
    # Assume all images have the same dimensions
    if len(images[0].shape) == 3:
        img_h, img_w, num_channels = images[0].shape
        grid_image = np.zeros((img_h * 3, img_w * 3, num_channels), dtype=images[0].dtype)
    else:
        img_h, img_w = images[0].shape
        grid_image = np.zeros((img_h * 3, img_w * 3), dtype=images[0].dtype)
        num_channels = None

    # Populate the grid image with the individual images
    for i in range(3):
        for j in range(3):
            img_index = i * 3 + j
            if img_index < len(images):  # Check if there are enough images
                if num_channels is not None:
                    grid_image[i * img_h: (i + 1) * img_h, j * img_w: (j + 1) * img_w, :] = images[img_index]
                else:
                    grid_image[i * img_h: (i + 1) * img_h, j * img_w: (j + 1) * img_w] = images[img_index]
    return grid_image

def verify_image(root_dir1, root_dir2, action, category1, category2):
    accept = [False]
    reject1 = [False]
    reject2 = [False]
    skip = [False]
    img_dir1 = os.path.join(root_dir1, "rgb_images")
    grid_image1 = make_grid_image(img_dir1)

    img_dir2 = os.path.join(root_dir2, "rgb_images")
    grid_image2 = make_grid_image(img_dir2)
    
    # Step 3: Display the combined grid image
    img_w, img_h = 490, 490
    fig, ax = plt.subplots(1, 2, figsize=(img_w * 6 / 180, img_h * 3.2 / 180))
    plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0)
    ax[0].imshow(grid_image1)
    ax[0].axis('off')  # Turn off axis
    ax[1].imshow(grid_image2)
    ax[1].axis('off')  # Turn off axis
    fig.suptitle(f'Action "{action}" using "{category1}" and "{category2}"', fontsize=18)

    # Horizontal lines between rows
    for i in range(0, 4):  # 2 horizontal lines between 3 rows
        ax[0].axhline(i * img_h - 0.5, color='gray', linewidth=1)
        ax[1].axhline(i * img_h - 0.5, color='gray', linewidth=1)

    # Vertical lines between columns
    for j in range(0, 4):  # 2 vertical lines between 3 columns
        ax[0].axvline(j * img_w - 0.5, color='gray', linewidth=1)
        ax[1].axvline(j * img_w - 0.5, color='gray', linewidth=1)

    ax_button = plt.axes([0.75, 0.94, 0.1, 0.05])  # Position of the button (left, bottom, width, height)
    accept_button = Button(ax_button, 'Accept')
    ax_button = plt.axes([0.87, 0.94, 0.1, 0.05])  # Position of the button (left, bottom, width, height)
    reject_button = Button(ax_button, 'Reject')
    def on_accept(event):
        accept[0] = True
        plt.close(fig)  # Close the plot

    def on_reject(event):
        accept[0] = False
        plt.close(fig)  # Close the plot

    accept_button.on_clicked(on_accept)
    reject_button.on_clicked(on_reject)

    ax_button = plt.axes([0.03, 0.94, 0.1, 0.05])  # Position of the button (left, bottom, width, height)
    reject1_button = Button(ax_button, 'Reject 1')
    ax_button = plt.axes([0.15, 0.94, 0.1, 0.05])  # Position of the button (left, bottom, width, height)
    reject2_button = Button(ax_button, 'Reject 2')

    def on_reject1(event):
        reject1[0] = True
        plt.close(fig)  # Close the plot
    def on_reject2(event):
        reject2[0] = True
        plt.close(fig)  # Close the plot
    
    reject1_button.on_clicked(on_reject1)
    reject2_button.on_clicked(on_reject2)

    ax_button = plt.axes([0.44, 0.88, 0.12, 0.05])  # Position of the button (left, bottom, width, height)
    skip_button = Button(ax_button, 'Skip Category Pair')
    def on_skip(event):
        skip[0] = True
        plt.close(fig)  # Close the plot
    skip_button.on_clicked(on_skip)

    plt.show()

    return reject1[0], reject2[0], accept[0], skip[0]

def main():
    mode = "seen"
    min_required = 50
    random.seed(42)

    with open("./metadata/objaverse_action2objects.json", "r") as f:
        action2objects = json.load(f)

    all_assets = os.listdir(f"objaverse_test/{mode}")
    os.makedirs(f"./lists_per_action/{mode}", exist_ok=True)

    if os.path.exists(f"./metadata/verify_pseudo_labels_{mode}.json"):
        with open(f"./metadata/verify_pseudo_labels_{mode}.json", "r") as f:
            verify_pseudo_labels = json.load(f)
    else:
        verify_pseudo_labels = {}
    if os.path.exists(f"./metadata/hard_rejects_{mode}.json"):
        with open(f"./metadata/hard_rejects_{mode}.json", "r") as f:
            hard_rejects = json.load(f)
    else:
        hard_rejects = {}
    if os.path.exists(f"./metadata/skip_catpairs_{mode}.json"):
        with open(f"./metadata/skip_catpairs_{mode}.json", "r") as f:
            skip_catpairs = json.load(f)
    else:
        skip_catpairs = {}

    all_actions = ['brush-with', 'cut-with', 'dig-with', 'press-with', 'lift-with', 'scrape-with', 'scoop-with', 'pick up-with', 'pierce-with', 'pour-with', 'poke-with', 'pound-with', 'roll-onto', 'write-with', 'sweep-with', 'hang-onto', 'pull-with', 'peel-with', 'apply torque-with', 'spread-with', 'wedge-with', 'mix-with', 'pry-with', 'sift-with']
    selected_actions = all_actions
    print(f"You are labelling {len(selected_actions)} actions: {selected_actions}.\n\n")

    for action, objects in action2objects.items():
        if action not in selected_actions:
            continue
        
        if action not in verify_pseudo_labels:
            verify_pseudo_labels[action] = {}
        if action not in hard_rejects:
            hard_rejects[action] = []
        if action not in skip_catpairs:
            skip_catpairs[action] = []

        action_assets = []
        for obj in objects:
            for x in all_assets:
                category = x.split("---")[-1].replace(".glb", "")
                if category == obj:
                    action_assets.append(x)
        
        category_combinations = list(itertools.combinations_with_replacement(objects, 2))
        random.shuffle(category_combinations)
        combinations = list(itertools.combinations(action_assets, 2))
        random.shuffle(combinations)
        selected = []

        print(f"\nProcessing action: {action}. Has {len(category_combinations)} category pairs and {len(combinations)} asset pairs.")
        
        for comb in combinations:
            if f"{comb[0]}|||{comb[1]}" in verify_pseudo_labels[action]:
                if verify_pseudo_labels[action][f"{comb[0]}|||{comb[1]}"]:
                    selected.append(comb)
            if f"{comb[1]}|||{comb[0]}" in verify_pseudo_labels[action]:
                if verify_pseudo_labels[action][f"{comb[1]}|||{comb[0]}"]:
                    selected.append((comb[1], comb[0]))
        
        for idx, category_combination in enumerate(category_combinations):
            if len(selected) >= min_required:
                break
            
            y1, y2 = category_combination
            # skip the category combination
            if [y1, y2] in skip_catpairs[action] or [y2, y1] in skip_catpairs[action]:
                continue

            selected_combinations = []
            for comb in combinations:
                x1, x2 = comb
                category1 = x1.split("---")[-1].replace(".glb", "")
                category2 = x2.split("---")[-1].replace(".glb", "")
                if (category1, category2) == category_combination or (category2, category1) == category_combination:
                    if category1 == category_combination[0]:
                        selected_combinations.append((x1,x2))
                    else:
                        selected_combinations.append((x2,x1))
            if len(selected_combinations) == 0:
                continue
            print(f"{idx+1}/{len(category_combinations)}: Selecting {category_combination} from {len(selected_combinations)} pairs")

            for comb in selected_combinations:
                if len(selected) >= min_required:
                    break

                x1, x2 = comb
                if x1 in hard_rejects[action] or x2 in hard_rejects[action]:
                    continue # contains an asset that's hard rejected
                if f"{x1}|||{x2}" in verify_pseudo_labels[action] or f"{x2}|||{x1}" in verify_pseudo_labels[action]:
                    continue
                
                category1 = x1.split("---")[-1].replace(".glb", "")
                category2 = x2.split("---")[-1].replace(".glb", "")
                reject1, reject2, accept, skip = verify_image(f"objaverse_test/{mode}/{x1}", f"objaverse_test/{mode}/{x2}", action, category1, category2)
                if skip: # skip the pair
                    break

                priority = None
                if reject1:
                    hard_rejects[action].append(x1)
                    priority = x2
                if reject2:
                    hard_rejects[action].append(x2)
                    priority = x1
                verify_pseudo_labels[action][f"{x1}|||{x2}"] = accept
                
                out_str = json.dumps(verify_pseudo_labels, indent=True)
                with open(f"./metadata/verify_pseudo_labels_{mode}.json", "w") as f:
                    f.writelines(out_str)
                out_str = json.dumps(hard_rejects, indent=True)
                with open(f"./metadata/hard_rejects_{mode}.json", "w") as f:
                    f.writelines(out_str)
                if verify_pseudo_labels[action][f"{x1}|||{x2}"]:
                    selected.append(comb)
                
                # prioritize the one not rejected
                while priority:
                    priority_combinations = [x for x in selected_combinations if priority in x]
                    new_priority = None
                    for priority_combination in priority_combinations:
                        if priority in hard_rejects[action]:
                            break
                        x1, x2 = priority_combination
                        if x1 in hard_rejects[action] or x2 in hard_rejects[action]:
                            continue # contains an asset that's hard rejected
                        if f"{x1}|||{x2}" in verify_pseudo_labels[action] or f"{x2}|||{x1}" in verify_pseudo_labels[action]:
                            continue
                        
                        category1 = x1.split("---")[-1].replace(".glb", "")
                        category2 = x2.split("---")[-1].replace(".glb", "")
                        if x1 == priority:
                            reject1, reject2, accept, skip = verify_image(f"objaverse_test/{mode}/{x1}", f"objaverse_test/{mode}/{x2}", action, category1, category2)
                        else:
                            reject2, reject1, accept, skip = verify_image(f"objaverse_test/{mode}/{x2}", f"objaverse_test/{mode}/{x1}", action, category2, category1)
                        if skip: # skip the pair
                            break
                        if reject1:
                            hard_rejects[action].append(x1)
                            if x1 == priority:
                                new_priority = x2
                        if reject2:
                            hard_rejects[action].append(x2)
                            if x2 == priority:
                                new_priority = x1
                        verify_pseudo_labels[action][f"{x1}|||{x2}"] = accept
                        
                        out_str = json.dumps(verify_pseudo_labels, indent=True)
                        with open(f"./metadata/verify_pseudo_labels_{mode}.json", "w") as f:
                            f.writelines(out_str)
                        out_str = json.dumps(hard_rejects, indent=True)
                        with open(f"./metadata/hard_rejects_{mode}.json", "w") as f:
                            f.writelines(out_str)
                        if verify_pseudo_labels[action][f"{x1}|||{x2}"]:
                            selected.append(priority_combination)
                            break
                    priority = new_priority

            skip_catpairs[action].append(category_combination)
            out_str = json.dumps(skip_catpairs, indent=True)
            with open(f"./metadata/skip_catpairs_{mode}.json", "w") as f:
                f.writelines(out_str)

        out_str = [f"{x[0]}, {x[1]}\n" for x in selected]
        with open(f"lists_per_action/{mode}/{action}.txt","w") as f:
            f.writelines(out_str)

if __name__ == "__main__":
    main()
