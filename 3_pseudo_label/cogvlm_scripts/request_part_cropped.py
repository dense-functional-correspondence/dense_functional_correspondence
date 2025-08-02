import requests
import re
import io
import os
import sys
import json
import glob
import base64
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw
from cogvlm_scripts.openai_api_request import create_chat_completion

def remove_image_extension(filename: str) -> str:
    # Define a regular expression pattern for common image file extensions
    pattern = r'\.(jpg|jpeg|png|gif|bmp|tiff)$'
    
    # Use re.sub to replace the pattern with an empty string if it's found at the end of the filename
    cleaned_filename = re.sub(pattern, '', filename, flags=re.IGNORECASE)
    
    return cleaned_filename

def encode_image(image):
    """
    Encodes a PIL image into a base64 string.
    """
    image_b = image_to_bytes(image)

    return base64.b64encode(image_b).decode("utf-8")

def image_to_bytes(image, format='JPEG'):
    # Create a BytesIO object
    img_byte_arr = io.BytesIO()
    
    # Save the image to the BytesIO object, using the specified format
    image.save(img_byte_arr, format=format)
    
    # Retrieve the byte data from the BytesIO object
    img_bytes = img_byte_arr.getvalue()
    
    return img_bytes

def square_pad(image):
    # Check the size of the image
    width, height = image.size
    
    # Determine the maximum of width and height
    max_side = max(width, height)
    
    # Create a new image with white background and size equal to the maximum dimension
    new_image = Image.new('RGB', (max_side, max_side), color='white')
    
    # Calculate the position to paste the original image onto the new image
    left = (max_side - width) // 2
    top = (max_side - height) // 2
    
    # Paste the original image onto the new image
    new_image.paste(image, (left, top))
    
    return new_image

def simple_image_chat(base_url, use_stream=True, query=None, img=None, text=None, top_p=0.8, temperature=0.9):
    img_url = f"data:image/jpeg;base64,{img}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        },
    ]
    output_str = create_chat_completion(
        "cogvlm-chat-17b", 
        base_url, 
        messages=messages, 
        use_stream=use_stream, 
        top_p=top_p, 
        temperature=temperature)
    return output_str

def generate_query(texts, template):
    res = []
    for text in texts:
        quoted_text = f'"{text[2:].strip()}"'
        # remove * and last . because of markdown formatting
        quoted_text = quoted_text.replace('*', '')
        quoted_text = quoted_text.replace('.', '')
        quoted_text = quoted_text.lower()
        res.append(template.replace("<expr>", quoted_text).strip() if template != "" else text.strip())
    return res

def process_cogvlm_bbox(coords, width, height):
    max_side = max(width, height)
    coords = np.array([float(coord) * 0.001 * max_side for coord in coords])
    pad_x = (max_side - width) // 2
    pad_y = (max_side - height) // 2
    coords[[0,2]] -= pad_x
    coords[[1,3]] -= pad_y
    coords = np.round(coords).astype(int)
    # force bbox to be within image
    coords[0] = max(coords[0], 0)
    coords[1] = max(coords[1], 0)
    coords[2] = min(coords[2], width)
    coords[3] = min(coords[3], height)
    coords = list(coords)
    coords = [int(coord) for coord in coords]
    return coords

def postprocess_image(text: str, img: Image) -> (str, Image):
    """
    Processes the given text to identify and draw bounding boxes on the provided image.
    This function searches for patterns in the text that represent coordinates for bounding
    boxes and draws rectangles on the image at these coordinates. Each box is drawn in a
    different color for distinction.
    Args:
        text (str): The text containing bounding box coordinates in a specific pattern.
        img (Image): The image on which to draw the bounding boxes.
    Returns:
        tuple[str, Image]: The processed text with additional annotations for each bounding
        box, and the image with the drawn bounding boxes.
    """
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]

    # Updated pattern to match single or multiple coordinate groups
    pattern = r"\[\[([\d,]+(?:;[\d,]+)*)\]\]"
    matches = re.findall(pattern, text)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if not matches:
        return text, None, None

    # well... we only really assume that there's one match and one bbox
    for i, match in enumerate(matches):
        # Splitting the matched string into individual coordinate groups
        coords_groups = match.split(';')

        # Determining the color for the current match
        color = colors[i % len(colors)]

        for coords_str in coords_groups:
            coords = coords_str.split(',')

            if len(coords) == 4:  # Rectangle
                processed_coords = process_cogvlm_bbox(coords, width, height)
                draw.rectangle(processed_coords, outline=color, width=3)
            else:
                return text, None, None

    return text, img, processed_coords

def get_responses(queries, img, ports, n_trials, pil_image, out_dir, viz_fn):
    query_to_bbox = {}
    for query_idx, query in enumerate(queries):
        query_to_bbox[f"query_{query_idx}"] = []
        success_trials = 0
        viz_img =  pil_image.copy()
        for i in range(n_trials * 2 // len(ports)): # max number of attempts
            # query multiple servers at once
            with ProcessPoolExecutor(max_workers=len(ports)) as executor:
                futures = [executor.submit(simple_image_chat, port, use_stream=False, query=query, img=img, top_p=0.95, temperature=0.9) for port in ports]
                for future in futures:
                    output_str = future.result()
                    try:
                        text, viz_img_new, bbox = postprocess_image(output_str, viz_img)
                        if viz_img_new is not None and bbox is not None:
                            query_to_bbox[f"query_{query_idx}"].append(bbox)
                            viz_img = viz_img_new
                            success_trials += 1
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue
                    if success_trials >= n_trials:
                        break
            if success_trials >= n_trials:
                break
        # if success_trials still less than n_trails, throw away the image!
        if success_trials < n_trials:
            return None
        img_output_f = os.path.join(out_dir, viz_fn+f"query_{query_idx}.png")
        viz_img.save(img_output_f)
    return query_to_bbox

def main(category, ports, orig_img_dir, out_dir, obj2part, n_trials):

    template_list = [
        "Where is <expr>? answer in [[x0,y0,x1,y1]] format.",
        "I'd like to know the exact bounding boxes of <expr> in the photo.",
        "What are the exact bounding boxes of <expr> in the provided picture?"]

    try:
        part2description = obj2part[category.split("---")[-1].replace(".glb","")]
    except:
        print(f"Category {category} does not exist!")
        return
    
    for part, description in part2description.items():
        img_dir = os.path.join(orig_img_dir, part.replace(" ", "_"))

        # if the directory doesn't exist, which means cropping in is not required, then skip 
        if not os.path.exists(img_dir):
            continue

        os.makedirs(os.path.join(out_dir, "part_annotations_cropped"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "part_viz_cogvlm_cropped"), exist_ok=True)
        image_files = glob.glob(f"{img_dir}/*")
        print(f"\nFinding masks for {part} of {category}.\nUsing the following descriptions:")
        out_dct = {}
        out_dct_f = os.path.join(out_dir, "part_annotations_cropped/"+part.replace(" ", "_")+".json")
        if os.path.exists(out_dct_f): # already ran
            continue

        # generate query
        if not isinstance(description, list):
            description = [description]
        queries = generate_query(description, template_list[2])
        for i, query in enumerate(queries):
            print(f"{i+1}. {query}")
        print("\n")

        for itr, image_path in enumerate(image_files):
            img_name = os.path.split(image_path)[-1]
            orig_img = Image.open(image_path).convert("RGB")
            encoded_img = encode_image(square_pad(orig_img))
            viz_fn = f"part_viz_cogvlm_cropped/{remove_image_extension(img_name)}-{part.replace(' ', '_')}-"
            query_to_bbox = get_responses(queries, encoded_img, ports, n_trials, orig_img, out_dir, viz_fn)
            if query_to_bbox is not None:
                out_dct[img_name]=query_to_bbox
        
        out_str = json.dumps(out_dct, indent=True)
        with open(out_dct_f, "w") as f:
            f.writelines(out_str)

def split_list(lst, n):
    avg_len = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

def run_main(args, categories, ports, obj2part, cluster_id):
    time.sleep(cluster_id*5) # hopefully avoid different processes writing at the same time...
    start_time = time.time()
    for category in categories:
        print(f"\nCluster {cluster_id+1}: Running object bbox detection on {category} images!")
        
        img_dir = os.path.join(args.root_dir, category, "rgb_images_cropped")
        out_dir = os.path.join(args.root_dir, category)
        main(category, ports, img_dir, out_dir, obj2part, args.n_trials)

        print(f"Cluster {cluster_id+1}: Finished {category} images. Running time: {time.time() - start_time} seconds.")
    time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('category', type=str, help="specific asset if all_category is False, otherwise ignored")
    parser.add_argument('--port', type=str, help="path to the .txt file that stores ip and port information")
    parser.add_argument('--root_dir', type=str, help="path to the data directory with rendered assets")
    parser.add_argument('--affordance_dir', type=str, help="object to part descriptions, e.g., obj2part_descriptions.json")
    parser.add_argument('--n_trials', type=int, default=4, help="number of CogVLM trials")
    parser.add_argument('--all_category', action='store_true', help='run on all categories')
    args = parser.parse_args()

    # ports
    with open(args.port, 'r') as file:
        ports = file.readlines()
    ports = [port.strip() for port in ports]
    ports = ["http://"+port.split(" ")[0]+":"+port.split(" ")[1] for port in ports]
    print(f"\nUsing {len(ports)} ports: {ports}.")
    
    num_cpus = min(os.cpu_count(), 8)
    # categories
    obj2part = json.load(open(args.affordance_dir))
    categories = []
    if args.all_category:
        for name in os.listdir(args.root_dir):
            if os.path.isdir(os.path.join(args.root_dir, name)) and name.split("---")[-1].replace(".glb","") in obj2part.keys():
                categories.append(name)
    else:
        for name in args.category.split(","):
            if os.path.isdir(os.path.join(args.root_dir, name)) and name.split("---")[-1].replace(".glb","") in obj2part.keys():
                categories.append(name)
    random.shuffle(categories)

    # divide into clusters.
    num_clusters = max(len(ports) // args.n_trials, 1)
    ports = split_list(ports, num_clusters)
    categories = split_list(categories, num_clusters)
    
    print(f"\nWe will split the data into {num_clusters} cluster(s).")
    for i in range(num_clusters):
        print(f"\nIn cluster {i+1}, we have {len(categories[i])} categories: {categories[i]}. And we will be using {len(ports[i])} ports: {ports[i]}.")

    # run main functions
    with ProcessPoolExecutor(max_workers=None) as main_executor:
        main_futures = [main_executor.submit(run_main, args, categories[i], ports[i], obj2part, i) for i in range(num_clusters)]

        for i, future in enumerate(as_completed(main_futures)):
            try:
                result = future.result()
                print(f"Cluster {i+1} finished with result: {result}")
            except Exception as e:
                print(f"Cluster {i+1} raised an exception: {e}")