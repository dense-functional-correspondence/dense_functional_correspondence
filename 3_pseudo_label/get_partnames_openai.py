import os
import json
import argparse
import sys
from openai import OpenAI

key = "<#TODO: your openai API key>"
os.environ['OPENAI_API_KEY'] = key

instructions = {
    "obj2obj":"Here are your instructions for the rest of the chat: Respond as if you are a human expert giving simplifying instructions to a robot learning to interact with the world by identifying object parts that correspond to verbs. We want to know what area of the object can be used to perform this action. We do not want to know the part that needs to be held to do this action. Respond with a list of part names, each with a sentence describing the part appearance in \"name - description\" format.",
    "hum2obj":"Here are your instructions for the rest of the chat: Respond as if you are a human expert giving simplifying instructions to a robot learning to interact with the world by identifying object parts that correspond to verbs. We want to know what area of the object a human would interact with to perform this action. Respond with a list of part names, each with a sentence describing the part appearance in \"name - description\" format."
    }

instructions_fewshot = {
    "obj2obj":"Here are your instructions for the rest of the chat: Respond as if you are a human expert giving simplifying instructions to a robot learning to interact with the world by identifying object parts that correspond to verbs. We want to know what area of the object can be used to perform this action. We do not want to know the part that needs to be held or interacted by a human to do this action. Respond with a list of part names, each with a sentence describing the part appearance in \"name - description\" format.\nWhen answering user questions, carefully consider the following 4 examples. Each example contains a question, a good answer, and a bad answer. The bad answers generally contain parts that the human explicitly interact with. Be sure to avoid bad answers.\n\nQuestion:\nWhat are the names of object parts of a \"knife\" that can be used to perform the action \"cut-with\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Blade - the flat, sharp part used for cutting.\n- Edge - The sharpened side of the blade that slices through materials.\nBad Answer:\n- Handle - the part where you grip the knife.\n\nQuestion:\nWhat are the names of object parts of a \"bottle\" that can be used to perform the action \"pour-with\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Mouth - The opening at the very top of the bottle from which liquids can be poured out.\n- Neck - The narrow part at the top of the bottle which tapers from the body leading to the opening.\nBad Answer:\n- Body - The cylindrical, central part of the bottle typically where labels are found.\n- Handle - A specifically designed part (if present) for easy gripping, typically protruding from the side or top of the bottle.\n\nQuestion:\nWhat are the names of object parts of a \"teapot\" that can be used to perform the action \"press-with\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Base - The bottom surface of the teapot, which can be pressed against a surface.\nBad Answer:\n- Grip - The part used for gripping, which could be pressed when held or manipulated.\n\nQuestion:\nWhat are the names of object parts of a \"water bottle\" that can be used to perform the action \"pound-with\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Body - The main cylindrical part of the bottle, which can be used to pound objects.\nBad Answer:\n- Neck - The narrow part between the body and cap, which could be gripped and used to pound with the bottle.",
    "hum2obj":"Here are your instructions for the rest of the chat: Respond as if you are a human expert giving simplifying instructions to a robot learning to interact with the world by identifying object parts that correspond to verbs. We want to know what area of the object a human would interact with to perform this action. Respond with a list of part names, each with a sentence describing the part appearance in \"name - description\" format.\nWhen answering user questions, carefully consider the following 3 examples. Each example contains a question, a good answer, and a bad answer. The bad answers generally contain parts that a human does NOT explicitly interact with. Be sure to avoid bad answers.\n\nQuestion:\nWhere would a human interact with a \"knife\" to perform the action \"cut\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Handle - the part where you grip the knife.\nBad Answer:\n- Blade - the flat, sharp part used for cutting.\n\nQuestion:\nWhere would a human interact with a \"bottle\" to perform the action \"pour\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Body - The cylindrical, central part of the bottle typically where labels are found.\n- Neck - The narrow part just below the cap, typically grasped or tilted while sipping.\nBad Answer:\n- Cap - The top part of the bottle that opens and closes.\n- Opening - The topmost part of the bottle that allows liquid to flow out.\n\nQuestion:\nWhere would a human interact with a \"baseball bat\" to perform the action \"hit\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Grip - The tapered lower end of the bat, where hands are placed to swing.\n- Handle - The narrow part at the end of the bat, typically covered with rubber or tape for grip.\nBad Answer:\n- Barrel - The thicker end of the bat, typically used for hitting the ball.\n\nQuestion:\nWhere would a human interact with a \"fork\" to perform the action \"stick\"? Respond with only a bulleted list of single word responses paired with short descriptions.\nGood Answer:\n- Handle - the part of the fork that is elongated and designed for grasping.\nBad Answer:\n- Tines - The pointed prongs at the end of the fork."
}

def get_obj2obj_q(obj_name, action):
    question = f"What are the names of object parts of a \"{obj_name}\" that can be used to perform the action \"{action}\"? Respond with only a bulleted list of single word responses paired with descriptions."
    return question

def get_hum2obj_q(obj_name, action):
    question = f"Where would a human interact with a \"{obj_name}\" to perform the action \"{action}\"? Respond with only a bulleted list of single word responses paired with short descriptions."
    return question

def get_responses(client, obj_action_dict, objects, mode="hum2obj", fewshot=False):
    assert mode in ["hum2obj", "obj2obj"]
    
    if mode == "hum2obj":
        q_func = get_hum2obj_q
    elif mode == "obj2obj":
        q_func = get_obj2obj_q

    output_dict = {obj:{} for obj in objects}
    
    count = 0
    total = sum([len(x) for x in obj_action_dict.values()])
    for obj_name, actions in obj_action_dict.items():
        print("\n", obj_name, actions)

        for action in actions:
            if fewshot:
                instruction = instructions_fewshot[mode]
            else:
                instruction = instructions[mode]

            question = q_func(obj_name, action)

            response = client.chat.completions.create(
              model="gpt-4-turbo",
              messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question},
              ]
            )

            output_str = response.choices[0].message.content
            output_list = output_str.split('\n')
            output_dict[obj_name][action] = output_list

            count+=1

            print(count, total, end='\r')

    return output_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object2action', type=str, help='object to action dictionary, e.g., 0_taxonomy_and_metadata/Objaverse/objaverse_object2actions.json')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--fewshot', action='store_true', help='use few-shot prompting or not')
    args = parser.parse_args()

    # HACK: always use fewshot!
    assert(args.fewshot == True)

    client = OpenAI()

    # obj2obj
    dct_p = args.obj2obj
    obj_action_dict = json.load(open(dct_p,"r"))
    objects = list(obj_action_dict.keys())
    obj2obj_output_dict = get_responses(client, obj_action_dict, objects, mode="obj2obj", fewshot=args.fewshot)
    obj2obj_out_str = json.dumps(obj2obj_output_dict, indent=True)

    if args.fewshot:
        with open(os.path.join(args.out_dir, "obj2obj_fewshot.json"), "w") as f:
            f.writelines(obj2obj_out_str)
    else:
        with open(os.path.join(args.out_dir, "obj2obj.json"), "w") as f:
            f.writelines(obj2obj_out_str)

if __name__ == "__main__":
    main()
