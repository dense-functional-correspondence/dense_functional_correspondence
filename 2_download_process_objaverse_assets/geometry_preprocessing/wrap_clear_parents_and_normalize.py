import os
import numpy as np
import subprocess
import glob
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

BLENDER = "<#TODO: path to your Blender>"
NUM_WORKERS = 8

def run_blender(src_f, out_f):
    # Define the command template with placeholders
    command_template = (
        f"{BLENDER} --background --python clear_parents_and_normalize.py -- "
        "--object_path {src_f} --output_path {out_f} "
    )

    # Fill in the placeholders with actual values
    command = command_template.format(
        src_f=src_f,
        out_f=out_f,
    )

    # Use shlex to split the command into a list suitable for subprocess
    command_args = shlex.split(command)

    # Run the command using subprocess
    try:
        subprocess.run(command_args, check=True, timeout=180)
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

    src_p = "<#TODO: path to the downloaded glbs>"
    dest_p = "<#TODO: path to the output normalized glbs>"

    os.makedirs(dest_p, exist_ok=True)

    items = glob.glob(f"{src_p}/*/*")

    already_done = os.listdir(dest_p)
    items = [x for x in items if x.split('/')[-1] not in already_done]

    commands = []

    for item in items:
        src_f = item
        out_f = os.path.join(dest_p, item.split("/")[-1])
        command = (src_f, out_f)
        commands.append(command)
    
    for i in range(0,len(commands), NUM_WORKERS):
            run_blender_parallel(commands[i:i+NUM_WORKERS])


if __name__ == "__main__":
    main() 