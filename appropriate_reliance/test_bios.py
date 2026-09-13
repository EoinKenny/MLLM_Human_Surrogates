import sys
import os
# Add parent directory to path so we can import the root llm_query module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import re
import ast

from sklearn.decomposition import PCA
from tqdm import tqdm

from llm_query import call_llm

def safe_call_llm(batch_prompts, model_name, temperature, batch_metadata=None,
                  max_retries=20, sleep_seconds=2):
    """
    Robust wrapper for call_llm that retries up to max_retries times.
    Returns a list of responses. If all retries fail, returns placeholders
    to preserve alignment with prompt_metadata.
    """
    for attempt in range(max_retries):
        try:
            return call_llm(batch_prompts, model_name=model_name,
                            temperature=temperature,
                            trace_metadata=batch_metadata)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                print(f"Batch failed after {max_retries} attempts: {e}. Continuing with empty responses.")
                return [""] * len(batch_prompts)

def get_profession(label: int) -> str:
    profession_map = {
        0: "accountant",
        1: "architect",
        2: "attorney",
        3: "chiropractor",
        4: "comedian",
        5: "composer",
        6: "dentist",
        7: "dietitian",
        8: "dj",
        9: "filmmaker",
        10: "interior_designer",
        11: "journalist",
        12: "model",
        13: "nurse",
        14: "painter",
        15: "paralegal",
        16: "pastor",
        17: "personal_trainer",
        18: "photographer",
        19: "physician",
        20: "poet",
        21: "professor",
        22: "psychologist",
        23: "rapper",
        24: "software_engineer",
        25: "surgeon",
        26: "teacher",
        27: "yoga_teacher"
    }
    return profession_map[label]

examples = [
    [
        "She has over 26 years experience in child development; child, adolescent and adult mental health and has worked extensively in government, community and hospital systems as well as private practice on the Gold Coast.",
        "He is also co-director of the Chippewa River Writing Project and vice-president of The Assembly for the Teaching of English Grammar (ATEG). Hyler has co-authored Create, Compose, Connect! Reading, Writing, and Learning with Digital Tools (Routledge, 2014) and From Texting to Teaching: Grammar Instruction in a Digital Age (Routledge, 2017) with Dr. Troy Hicks. Follow him @Jeremybballer",
        "The primary focus of his work has been on urbanization and planning in cities of developing countries, with particular emphasis on Asian cities. Since the time of his doctoral research on land development in Jakarta, Indonesia (PhD Berkeley, 1992), Dr. Leaf has been extensively involved in urbanization research and capacity building projects in Indonesia, Vietnam, China, Thailand and Sri Lanka. The courses he teaches at SCARP cover the theory and practices of development planning and the social, institutional and environmental aspects of urbanization in developing countries.",
        "A Colorado native, she grew up in Grand Junction and attended the University of Colorado at Boulder. She obtained her Master's degree from Rosalind Franklin University of Medicine and Science in North Chicago, IL, and then returned to Colorado to practice. Ms. Richardson chose to be a PA in family medicine in order to develop long-standing relationships with her patients. She especially enjoys preventive care and women's health. Outside of medicine, her interests include camping, rafting, snowboarding and traveling.",
        "He is the founding spiritual director of the Medicine Buddha Tantrayana Meditation Centre, the Tibetan Children's Fund and the Shantideva Buddhist Foundation Ltd. Lama Tendar travels across Australia to teach Buddhism, meditation, perform blessing ceremonies and healing. In 2015 he was the Chant Master for His Holiness the Dalai Lama's Australian tour in the Blue Mountains and Brisbane."
    ],
    ['psychologist', 'teacher', 'professor', 'physician', 'teacher']
]

class_names = ['teacher', 'psychologist', 'surgeon', 'professor', 'physician']

def get_llm_label(s):
    """
    Use regex to find a single word from the list ['teacher', 'psychologist', 'surgeon', 'professor', 'physician']
    between the <answer> tags, allowing for white space.
    """
    pattern = r'<answer>\s*(teacher|psychologist|surgeon|professor|physician)\s*</answer>'
    match = re.search(pattern, s, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

def round_numbers_in_string(data_string):
    # numpy>=2 reprs scalars as np.float64(0.12) / np.str_('word'), which
    # ast.literal_eval rejects (it sees a function call). Strip the wrapper.
    cleaned = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", str(data_string))
    data_list = ast.literal_eval(cleaned)
    rounded_list = [(str(item[0]), round(float(item[1]), 4)) for item in data_list]
    rounded_string = str(rounded_list)
    return rounded_string

def make_prompt_control(row, examples):
    prompt = f"""
    Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a
    study to examine people's reliance on AI systems with or without explanations for the AI's predictions.

    The domain is profession classification based on a sequence of text.

    The possible classes are: {class_names}

    Here are 5 randomly sampled examples of training data to help you understand the domain: {  examples[0]  },
    and here are their labels: {  examples[1]  }

    Your task is to look at a new test instance, and decide what you think the real label is.

    Here is the new test instance: {row.test_instance}

            Use only the supplied information. Choose the best supported option even when the biography is incomplete.

    Outline your reasoning process step by step, and give your final answer in the tags <answer>...</answer>.
    """
    return prompt

def make_prompt_saliency(row, examples):
    if row.explanation_target_label != row.test_predicted_label:
        raise ValueError("Saliency target does not match displayed prediction")
    prompt = f"""
    Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a
    study to examine people's reliance on AI systems with or without explanations for the AI's predictions.

    The domain is profession classification based on a sequence of text.

    The possible classes are: {class_names}

    Here are 5 randomly sampled examples of training data to help you understand the domain: {  examples[0]  },
    and here are their labels: {  examples[1]  }

    Your task is to look at a new test instance, and an AI's prediction for its label, and decide what you think the real label is.

    Here is the new test instance: {row.test_instance}

            Use only the supplied information. Choose the best supported option even when the biography is incomplete.

    Here is the AI prediction for the new test instance: {  row.test_predicted_label  }

    In addition, here is a feature saliency explanation for the AI prediction where you see each feature, 
    and its contribution (positive or negative) towards the prediction: {round_numbers_in_string(row.lime_explanation)}    

    Use only the supplied information. The AI prediction may be incorrect.

    Outline your reasoning process step by step, and give your final answer in the tags <answer>...</answer>.
    """
    return prompt

def make_prompt_nns(row, examples):
    prompt = f"""
    Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a
    study to examine people's reliance on AI systems with or without explanations for the AI's predictions.

    The domain is profession classification based on a sequence of text.

    The possible classes are: {class_names}

    Here are 5 randomly sampled examples of training data to help you understand the domain: {  examples[0]  },
    and here are their labels: {  examples[1]  }

    Your task is to look at a new test instance, and an AI's prediction for its label, and decide what you think the real label is.

    Here is the new test instance: {row.test_instance}

            Use only the supplied information. Choose the best supported option even when the biography is incomplete.

    Here is the AI prediction for the new test instance: {  row.test_predicted_label  }

    In addition, here is an explanation for the prediction in the form of the two nearest neighbors from the training data
    that the AI learned from: 

    The first nearest neighbor is {row.neighbor1_instance},
    its label is '{  row.neighbor1_true_label }',
    and the AI's prediction of this nearest neighbor is '{  row.neighbor1_predicted_label }'
    The second nearest neighbor is {row.neighbor2_instance},
    its label is '{  row.neighbor2_true_label }',
    and the AI's prediction of this nearest neighbor is '{  row.neighbor2_predicted_label }'

    Use only the supplied information. The AI prediction may be incorrect.

    Outline your reasoning process step by step, and give your final answer in the tags <answer>...</answer>.
    """
    return prompt

prompt_mapping = {
    "control": make_prompt_control,
    "saliency": make_prompt_saliency,
    "nns": make_prompt_nns
}

if __name__ == "__main__":
    # Configuration via argparse
    parser = argparse.ArgumentParser(description="Run bios reliance experiment with configurable model, batch size, and temperature.")
    parser.add_argument("--model_name", type=str, default="qwen3-vl-8b", help="Specify model name to use")
    parser.add_argument("--batch_size", type=int, default=50, help="Specify batch size for processing prompts")
    parser.add_argument("--temp", type=float, default=1.0, help="Specify temperature for LLM sampling")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sanity check: only use the first N test instances per seed (default: all)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds to run, e.g. '0' or '0,1'. Default: 0-25")
    args = parser.parse_args()

    temp = args.temp
    batch_size = args.batch_size
    model_name = args.model_name
    limit = args.limit
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(26))
    num_repetitions = 1  # One response per test instance/strategy for this run

    for seed in seeds:

        df = pd.read_csv("data_v2/bios_neighbors_and_explanations_seed_"+str(seed)+".csv")

        if limit is not None:
            df = df.head(limit).copy()
            print(f"[sanity] seed {seed}: limited to first {len(df)} test instance(s)")
        
        # Shared definitions are at module scope.

        # --- Hard-coded examples ---
        # Shared definitions are at module scope.
        print("Examples:")
        print(examples)

        # Shared definitions are at module scope.

        # Shared definitions are at module scope.

        # Shared definitions are at module scope.

        # Shared definitions are at module scope.

        # Shared definitions are at module scope.

        # Shared definitions are at module scope.

        # Shared definitions are at module scope.
        
        for key in prompt_mapping.keys():
            df[f'{key}_labels'] = None
            df[f'{key}_texts'] = None

        # Step 1: Gather all prompts
        print("Gathering all prompts...")
        all_prompts = []
        prompt_metadata = []  # Store (row_idx, prompt_type, repetition_idx)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating prompts"):
            for key, prompt_func in prompt_mapping.items():
                for rep in range(num_repetitions):
                    llm_prompt = prompt_func(row, examples)
                    all_prompts.append(llm_prompt)
                    prompt_metadata.append((idx, key, rep))
        
        print(f"Total prompts to process: {len(all_prompts)}")
        
        # Step 2: Process prompts in batches
        print(f"Processing prompts in batches of {batch_size}...")
        all_responses = []
        
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Processing batches"):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_meta = [
                {"study": "bios", "instance_id": f"seed:{seed}:row:{idx}:rep:{rep}", "condition": key,
                 "repetition": rep, "seed_index": seed}
                for idx, key, rep in prompt_metadata[i:i+batch_size]
            ]
            batch_responses = safe_call_llm(
                batch_prompts, model_name=model_name, temperature=temp,
                batch_metadata=batch_meta)
            all_responses.extend(batch_responses)
        
        print(f"Total responses received: {len(all_responses)}")
        
        # Step 3: Parse responses and populate dataframe
        print("Parsing responses and populating dataframe...")
        
        # Initialize storage for results
        results_storage = {}
        for idx in df.index:
            results_storage[idx] = {}
            for key in prompt_mapping.keys():
                results_storage[idx][f'{key}_labels'] = []
                results_storage[idx][f'{key}_texts'] = []
        
        # Process all responses
        for response, (idx, key, rep) in zip(all_responses, prompt_metadata):
            prediction = get_llm_label(response)
            results_storage[idx][f'{key}_labels'].append(prediction)
            results_storage[idx][f'{key}_texts'].append(response)
        
        # Populate dataframe
        for idx in df.index:
            for key in prompt_mapping.keys():
                df.at[idx, f'{key}_labels'] = results_storage[idx][f'{key}_labels']
                df.at[idx, f'{key}_texts'] = results_storage[idx][f'{key}_texts']
        
        # Save the dataframe after processing all rows
        out_dir = "data/reliance/bios_results"
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_limit{limit}" if limit is not None else ""
        df.to_csv(
            os.path.join(
                out_dir,
                f"results_bios_seed_{seed}_{model_name}{suffix}.csv",
            ),
            index=False
        )
        print(f"Saved results for seed {seed}")
        
        
        
        
        
        
