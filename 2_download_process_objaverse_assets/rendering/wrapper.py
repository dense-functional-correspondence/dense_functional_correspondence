import os
import json
import shlex
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import argparse
from collections import defaultdict

BLENDER = "<#TODO: path to your Blender>"
SCRIPT = "render_glb.py"
NUM_WORKERS = 8


def run_blender(obj_file, hdr_file, output_dir, ico_density, width, height):
    # Define the command template with placeholders
    command_template = (
        f"{BLENDER} --background --python {SCRIPT} -- "
        "'{obj_file}' '{hdr_file}' '{output_dir}' {ico_density} {width} {height}"
    )

    # Fill in the placeholders with actual values
    command = command_template.format(
        obj_file=obj_file,
        hdr_file=hdr_file,
        output_dir=output_dir,
        ico_density=ico_density,
        width=width,
        height=height
    )

    # Use shlex to split the command into a list suitable for subprocess
    command_args = shlex.split(command)

# Run the command using subprocess
    try:
        subprocess.run(command_args, check=True)
        print("Blender command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Blender command execution failed.")

def run_blender_parallel(commands):
    # Use ThreadPoolExecutor to run commands in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_blender, *cmd) for cmd in commands]

        # Optionally, you can handle results and exceptions
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exception that occurred in the thread
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--start",
        type=int,
        help="The starting integer"
    )

    parser.add_argument(
        "--end",
        type=int,
        help="The ending integer"
    )

    args = parser.parse_args()
    mesh_path = "<#TODO: path to normalized glbs>"
    render_path = "<#TODO: path to output rendered images>"
    hdr_path = "../../0_taxonomy_and_metadata/hdris/hdr"
    hdr_files = glob.glob(f"{hdr_path}/*")
    assets_json = "../../0_taxonomy_and_metadata/Objaverse/verified_assets.json"
    assets_dct = json.load(open(assets_json))
    
    items = []
    for category, assets in assets_dct.items():
        items.extend([(category, x) for x in assets])
    
    # default to render everything
    if args.start is None:
        args.start = 0
    if args.end is None:
        args.end = len(items)
    items = items[args.start:args.end]

    print(f"Generating {len(items)} items from start {args.start} to {args.end}")

    ico_density = 2
    width = 490
    height = 490

    commands = []

    for item in items:
        hdr_file = random.choice(hdr_files)
        categ, obj_id = item 
        obj_file = os.path.join(mesh_path, obj_id+'.glb')
        name = f"{obj_id}---{categ}.glb"
        output_dir = os.path.join(render_path, name)

        command = (obj_file, hdr_file, output_dir, ico_density, width, height)
        commands.append(command)

    for i in range(0,len(commands), NUM_WORKERS):
        run_blender_parallel(commands[i:i+NUM_WORKERS])

if __name__ == "__main__":
    main()
