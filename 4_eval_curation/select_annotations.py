import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import matplotlib.patches as patches
from PIL import Image

mode = "seen"
N_TRIALS = 6
ANNO_PATH = f"./labeled_transforms/{mode}"
VIZ_PATH = f"./2d_annotations_viz/{mode}"  # output by 3d_to_2d.py


def combine_images(first_image, second_image):
    first_image = Image.open(first_image).convert("RGBA")
    first_image_rgb = Image.new("RGB", first_image.size, (255, 255, 255))  # Create a white background
    first_image_rgb.paste(first_image, mask=first_image.split()[3])
    first_image = first_image_rgb
    second_image = Image.open(second_image).convert("RGB")

    cropped_image = second_image.crop((55, 99, 625, 382))
    resized_image = cropped_image.resize((448, 224))

    combined_width = first_image.width + resized_image.width
    combined_height = first_image.height
    combined_image = Image.new("RGB", (combined_width, combined_height))

    combined_image.paste(first_image, (0, 0))
    combined_image.paste(resized_image, (first_image.width, 0))
    return combined_image

def pick_image(image_paths, action, suffix):
    # Load the images from file paths
    ind_images = [os.path.join(VIZ_PATH, img_path) for img_path in image_paths]
    images = []
    for i in range(0, len(ind_images), 2):
        images.append(np.array(combine_images(ind_images[i], ind_images[i+1])))

    # Step 2: Preprocess the images into a single grid
    img_h, img_w, num_channels = images[0].shape

    # Create an empty array for the grid image
    grid_image = np.ones((img_h * 3, img_w * 2, num_channels), dtype=images[0].dtype) * 255

    # Populate the grid image with the individual images
    for i in range(N_TRIALS):
        for j in range(2):
            img_index = i * 2 + j
            if img_index < len(images):  # Check if there are enough images
                try:
                    grid_image[i * img_h: (i + 1) * img_h, j * img_w: (j + 1) * img_w, :] = images[img_index]
                except:
                    print("Something seems to be wrong? skip...")
    # Step 3: Display the combined grid image
    fig, ax = plt.subplots(figsize=(img_w * 2 / 100, img_h * 3 / 100))
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0)
    ax.imshow(grid_image)
    ax.set_title(f"Pick the views to keep for action '{action}' {suffix}", fontsize=20)
    ax.axis('off')  # Turn off axis

    # Horizontal lines between rows
    for i in range(1, 3): 
        ax.axhline(i * img_h - 2, color='black', linewidth=4)
    # Vertical lines between columns
    for j in range(1, 2):
        ax.axvline(j * img_w - 2, color='black', linewidth=4)

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
                image_index = row * 2 + col  # The index of the clicked image in the 4x6 grid
                if image_index < len(image_paths) and image_index not in selected_image_indices:
                    selected_image_indices.append(image_index)  # Add the index to the list
                    
                    # Add rectangle
                    rect = patches.Rectangle((col * img_w+2, row * img_h+2), img_w-6, img_h-6,
                                            edgecolor='red', facecolor='none', linewidth=4)
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
    return selected_image_indices

if __name__ == "__main__":
    actions = sorted(os.listdir(ANNO_PATH))
    actions = [x for x in actions if x[0]!="."]
    all_viz_files = os.listdir(VIZ_PATH)
    all_viz_files = [x for x in all_viz_files if x[0]!="."]

    if os.path.exists(f"./selected_annotations_{mode}.json"):
        selected_annotations = json.load(open(f"./selected_annotations_{mode}.json"))
    else:
        selected_annotations = {}

    for action in actions:
        print(f"Processing action: {action}")
        if action not in selected_annotations:
            selected_annotations[action] = {}
        obj_pairs = sorted(os.listdir(f"{ANNO_PATH}/{action}"))
        obj_pairs = [x for x in obj_pairs if x[0]!="."]

        for i, obj_pair in enumerate(obj_pairs):
            if obj_pair in selected_annotations[action]: # already processed
                continue
            
            obj1 = obj_pair.split("|||")[0]
            obj2 = obj_pair.split("|||")[1]
            relevant_viz_files = [x for x in all_viz_files if action in x and obj1 in x and obj2 in x]
            relevant_viz_files_ordered = []
            for trial in range(N_TRIALS):
                relevant_viz_files_ordered += [x for x in relevant_viz_files if f"trial_{trial}" in x and "line" not in x]
                relevant_viz_files_ordered += [x for x in relevant_viz_files if f"trial_{trial}" in x and "line" in x]
            suffix = f"{i+1}/{len(obj_pairs)}"
            selected_trials = pick_image(relevant_viz_files_ordered, action, suffix)
            
            selected_annotations[action][obj_pair] = selected_trials
            out_str = json.dumps(selected_annotations, indent=True)
            with open(f"./selected_annotations_{mode}.json", "w") as f:
                f.writelines(out_str)
