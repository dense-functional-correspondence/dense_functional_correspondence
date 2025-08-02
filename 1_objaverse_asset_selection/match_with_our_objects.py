import json
import pandas as pd
import ollama
import time
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['OLLAMA_MODELS'] = "<#TODO: path to your llama models>"
# https://ollama.com/blog/embedding-models
ollama.pull('mxbai-embed-large')

pred_categories = json.load(open("llm_summarized_objaverse_categories.json"))

OUTPUT_PATH = "objaverse_matched_categories.json"


def calc_cosine_similarity(embedding, referece_embeds):
    return np.dot(referece_embeds, embedding) / (np.linalg.norm(embedding) * np.linalg.norm(referece_embeds, axis=1))

action2objects = json.load(open("../0_taxonomy_and_metadata/action2objects.json"))
all_objects = []
for action, objects in action2objects.items():
    all_objects += objects
all_objects = list(set(all_objects))
print(f"\nIn our taxonomy, there are {len(all_objects)} unique object names.")

all_objects_embeddings = []
for obj in all_objects:
    output = ollama.embeddings(
      model='mxbai-embed-large',
      prompt=obj,
    )
    all_objects_embeddings.append(np.array(output["embedding"]))  # 1024 dimensional
all_objects_embeddings = np.array(all_objects_embeddings)
print(f"Reference objects embeddings have shape {all_objects_embeddings.shape}.")

result = []
for name, pred_category in tqdm(pred_categories.items(), total=len(pred_categories), desc=f"Matching:", ncols=80):

    pred_category = pred_category.replace("\"","").lower()
    output = ollama.embeddings(
      model='mxbai-embed-large',
      prompt=pred_category,
    )
    embedding = output["embedding"]

    cosine_sim = calc_cosine_similarity(embedding, all_objects_embeddings)
    most_similar_indices = np.argsort(cosine_sim)[::-1]
    most_similar_index = most_similar_indices[0]
    if cosine_sim[most_similar_index] >= 0.7:
        result.append([name, pred_category, all_objects[most_similar_index], str(cosine_sim[most_similar_index])])

    if len(result) % 1000 == 0:
        out_str = json.dumps(result, indent=True)
        with open(OUTPUT_PATH, "w") as f:
                f.writelines(out_str)
            
out_str = json.dumps(result, indent=True)
with open(OUTPUT_PATH, "w") as f:
        f.writelines(out_str)
