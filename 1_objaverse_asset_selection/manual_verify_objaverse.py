import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import matplotlib.patches as patches

# NOTE: run this on local computer. 
# Put renders for each asset inside "manual_verification_renders" folder, and add objaverse_action2objects.json
# After verification, confirmed assets will be saved to verified_assets.json


def pick_image(image_paths, category):
    # Load the images from file paths
    images = [mpimg.imread(img_path) for img_path in image_paths]

    # Step 2: Preprocess the images into a single grid (4x6)
    img_h, img_w, num_channels = images[0].shape

    # Create an empty array for the grid image (4 rows, 6 columns)
    grid_image = np.zeros((img_h * 4, img_w * 6, num_channels), dtype=images[0].dtype)

    # Populate the grid image with the individual images
    for i in range(4):
        for j in range(6):
            img_index = i * 6 + j
            if img_index < len(images):  # Check if there are enough images
                grid_image[i * img_h: (i + 1) * img_h, j * img_w: (j + 1) * img_w, :] = images[img_index]

    # Step 3: Display the combined grid image
    fig, ax = plt.subplots(figsize=(img_w * 6 / 190, img_h * 4 / 190))
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0)
    ax.imshow(grid_image)
    ax.set_title(f"Pick the assets to discard for '{category}'", fontsize=20)
    ax.axis('off')  # Turn off axis

    # Horizontal lines between rows
    for i in range(1, 4):  # 3 horizontal lines between 4 rows
        ax.axhline(i * img_h - 2, color='white', linewidth=4)

    # Vertical lines between columns
    for j in range(1, 6):  # 5 vertical lines between 6 columns
        ax.axvline(j * img_w - 2, color='white', linewidth=4)

    selected_image_indices = []  # List to store selected image indices

    def onclick(event):
        # Get the click coordinates relative to the grid image
        x = event.xdata
        y = event.ydata
        
        if x is not None and y is not None:
            if int(x) == 0 and int(y) == 0:
                pass
            else:
                # Calculate the row and column of the clicked image in the grid
                col = int(x // img_w)
                row = int(y // img_h)

                # Calculate the image index
                image_index = row * 6 + col  # The index of the clicked image in the 4x6 grid
                if image_index < len(image_paths) and image_index not in selected_image_indices:
                    selected_image_indices.append(image_index)  # Add the index to the list
                    
                    # Add rectangle
                    rect = patches.Rectangle((col * img_w+2, row * img_h+2), img_w-8, img_h-8,
                                            edgecolor='red', facecolor='none', linewidth=6)
                    ax.add_patch(rect)
                    # Use blitting for faster updates
                    ax.draw_artist(rect)
                    fig.canvas.blit(ax.bbox)  # Update only the axes area

    # Create a button on the top right to finish selection and disconnect
    ax_button = plt.axes([0.8, 0.94, 0.1, 0.05])  # Position of the button (left, bottom, width, height)
    button = Button(ax_button, 'Finish')

    def on_finish(event):
        fig.canvas.mpl_disconnect(cid)  # Disconnect the click event
        plt.close(fig)  # Close the plot

    button.on_clicked(on_finish)

    # Step 5: Register the click event and display the plot
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Return the selected image paths after the plot is closed
    return [image_paths[i] for i in selected_image_indices] if selected_image_indices else []


if __name__ == "__main__":
    base_dir = "./manual_verification_renders"
    all_categories = os.listdir(base_dir)
    all_categories = [x for x in all_categories if x[0] != "."] # avoid .DS_Store
    
    action2objects = json.load(open("./objaverse_action2objects.json"))

    if os.path.exists("./verified_assets.json"):
        picked_assets = json.load(open("./verified_assets.json"))
    else:
        picked_assets = {}

    for idx, category in enumerate(all_categories):
        if category in picked_assets:
            continue
        else:
            picked_assets[category] = []

        actions = []
        for action, objects in action2objects.items():
            if category in objects:
                actions.append(action)

        category_dir = os.path.join(base_dir, category)
        all_images = os.listdir(category_dir)
        all_images = [x for x in all_images if x[0] != "."] # avoid .DS_Store

        print(f"\n{idx+1}/{len(all_categories)} {category} ({len(all_images)}): {actions}")

        chunk_size = 24
        max_num_assets = 200
        for i in range(0, len(all_images), chunk_size):
            chunk = all_images[i:i + chunk_size]
            image_paths = [os.path.join(category_dir, x) for x in chunk]
            images_to_discard = pick_image(image_paths, category)
            images_to_discard = [x.split("/")[-1] for x in images_to_discard]
            for img in chunk:
                if img not in images_to_discard:
                    picked_assets[category].append(img.replace(".png", ""))
                if len(picked_assets[category]) >= max_num_assets:
                    break
            
            if len(picked_assets[category]) >= max_num_assets:
                break
            
            # save
            out_str = json.dumps(picked_assets, indent=True)
            with open("./verified_assets.json", "w") as f:
                f.writelines(out_str)
        
        # save
        out_str = json.dumps(picked_assets, indent=True)
        with open("./verified_assets.json", "w") as f:
            f.writelines(out_str)
    
    out_str = json.dumps(picked_assets, indent=True)
    with open("./verified_assets.json", "w") as f:
        f.writelines(out_str)
        
            

        