import os
import json
import sys
import argparse
from openai import OpenAI

key = "<#TODO: your openai API key>"
os.environ['OPENAI_API_KEY'] = key

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj2action2part', type=str, help="The output of get_partnames_openai.py")
    args = parser.parse_args()

    with open(args.obj2action2part) as f:
        anno = json.load(f)

    obj2part = {}
    unique_part_count = 0
    total_part_count = 0
    for obj, action2part in anno.items():
        obj2part[obj] = {}
        for action, parts in action2part.items():
            for part in parts:
                if part[0] != "-":
                    print(f"ERROR: {obj} with action {action} has bad part names!")
                    sys.exit()
                part_processed = part[2:].strip().replace('*', '')
                part_name = part_processed.split("- ")[0].strip().lower()
                if part_name not in obj2part[obj].keys():
                    obj2part[obj][part_name] = []
                obj2part[obj][part_name].append(part.replace(u"\u2018", "'").replace(u"\u2019", "'"))
                total_part_count += 1
        unique_part_count += len(obj2part[obj])
    
    print(f"There are {total_part_count} part descriptions in total, but {unique_part_count} unique parts.")

    obj2part_unique = obj2part
    obj2part = json.dumps(obj2part, indent=True)
    with open(os.path.join(os.path.dirname(args.obj2action2part), "obj2part.json"), "w") as f:
        f.writelines(obj2part)

    # Now, get the unique list!
    client = OpenAI()
    instruction = "Here are your instructions for the rest of the chat: Respond as if you are a human expert providing descriptions for object parts. For an object, you will be given a bulleted list containing the same part name with slightly different descriptions. Your task is to synthesize and summarize these into a single, one-line description in the format of \"- {part name} - {description}\". Ensure the description is descriptive and unambiguous but also very concise, avoiding redundancy."
    
    i = 0
    for obj, part2description in obj2part_unique.items():
        for part, description in part2description.items():
            i+=1
            if len(description) > 1:  # more than 1 descriptions
                question = f"The object is \"{obj}\" and the part is \"{part}\". Below are the part descriptions. Synthesize and summarize into a one-line description in the format of " + "\"- {part name} - {description}\".\n"
                for desc in description:
                    question += desc
                    question += "\n"
                response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": question},
                        ]
                        )
                output_str = response.choices[0].message.content
                print(f"\n{i}/{unique_part_count}: The object is \"{obj}\" and the part is \"{part}\". The synthesized description is:\n{output_str}\nApprove or not? If not, provide a user-written string.")
                user_input = input("Your answer: ")
                if user_input == "" or user_input.lower() == "y" or user_input.lower() == "yes":
                    obj2part_unique[obj][part] = output_str
                else: 
                    obj2part_unique[obj][part] = user_input
            else:
                obj2part_unique[obj][part] = description[0]

    obj2part_unique = json.dumps(obj2part_unique, indent=True)
    with open(os.path.join(os.path.dirname(args.obj2action2part), "obj2part_descriptions.json"), "w") as f:
        f.writelines(obj2part_unique)
