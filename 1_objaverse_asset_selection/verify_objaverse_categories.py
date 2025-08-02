import json
import pandas as pd
import ollama
import time
from tqdm import tqdm
import os

os.environ['OLLAMA_MODELS'] = "<#TODO: path to your llama models>"
ollama.pull('llama3.1:8b-instruct-fp16')

# Download caption 3D annotations from https://huggingface.co/datasets/tiange/Cap3D
captions = pd.read_csv("<#TODO: path to Caption3D annotation>", header=None)
captions = captions.set_index(0)[1].to_dict()

matches = json.load(open("objaverse_matched_categories.json"))

OUTPUT_PATH = "objaverse_llm_verified_categories.json"


system_prompt = "You will be given a description of a 3D asset and an object name. Your task is to determine if the description accurately matches the name. Respond only with 'yes' if they match, or 'no' if they do not. Output only 'yes' or 'no'."
prompt = "Description: '<cap>'\nObject Name: '<name>'"

result = []
success_match = 0
for entry in tqdm(matches,  total=len(matches), desc="Verifying", ncols=80):
    key, _, object_name, similarity = entry
    sha = key
    caption = captions[sha]

    num_trials = 0
    while num_trials < 5:
        response = ollama.chat(model='llama3.1:8b-instruct-fp16', messages=[                                                      
            {
            'role': 'system',
            'content': system_prompt,
            },
            {
            'role': 'user',                        
            'content': prompt.replace("<cap>", caption).replace("<name>", object_name),
            },                                     
        ])  
        num_trials += 1

        response = response['message']['content']
        response = response.strip().replace("\"","").replace("\'","").replace(".","").lower()
        if response == "yes" or response == "no":
            break

    if response == "yes":
        result.append([key, object_name])
        success_match += 1
        
    if len(result) % 100 == 0:
        out_str = json.dumps(result, indent=True)
        with open(OUTPUT_PATH, "w") as f:
                f.writelines(out_str)

out_str = json.dumps(result, indent=True)
with open(OUTPUT_PATH, "w") as f:
        f.writelines(out_str)
print(f"There are {success_match} successful final matches in Caption3D.")
