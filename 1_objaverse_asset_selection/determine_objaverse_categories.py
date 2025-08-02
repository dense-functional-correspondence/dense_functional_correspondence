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

OUTPUT_PATH = "llm_summarized_objaverse_categories.json"


system_prompt = "You will be provided with a brief caption or description of a 3D asset. Your task is to generate the most concise, accurate, and contextually appropriate object name based on the given description. The object name should reflect the core identity of the asset, avoiding overly specific labels. Output only the object name."
prompt = "The caption is '<cap>'. Based on this description, provide the most fitting and concise object name."

result = {}
for sha, caption in tqdm(captions.items(), total=len(captions), desc="Processing", ncols=80):
    caption = captions[sha]
    response = ollama.chat(model='llama3.1:8b-instruct-fp16', messages=[                                                      
      {
        'role': 'system',
        'content': system_prompt,
      },
      {
        'role': 'user',                        
        'content': prompt.replace("<cap>", caption),
      },                                     
    ])  
    result[sha] = response['message']['content']
    if len(result) % 1000 == 0:
        out_str = json.dumps(result, indent=True)
        with open(OUTPUT_PATH, "w") as f:
            f.writelines(out_str)

out_str = json.dumps(result, indent=True)
with open(OUTPUT_PATH, "w") as f:
    f.writelines(out_str)
