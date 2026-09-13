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

feature_names = ['age', 'workclass', 'education', 'marital-status', 'occupation',
        'native-country', 'hours-per-week', 'gender', 'race']

class_names = ['earns > 50k', 'earns < 50k']


PROMPT_VERSION = "adult-v2-paired-demonstrations"
ADULT_EXAMPLES = [[37, 'Self-emp-inc', 'HS-grad', 'Married-civ-spouse', 'Sales', 'United-States', 60, 'Male', 'White'], [56, 'Private', 'Some-college', 'Married-civ-spouse', 'Craft-repair', 'United-States', 40, 'Male', 'White'], [61, 'Private', 'HS-grad', 'Divorced', 'Tech-support', 'United-States', 40, 'Female', 'White'], [36, 'Self-emp-not-inc', 'Bachelors', 'Married-civ-spouse', 'Farming-fishing', 'United-States', 50, 'Male', 'White'], [35, 'Private', 'HS-grad', 'Married-civ-spouse', 'Machine-op-inspct', 'United-States', 45, 'Male', 'White'], ['earns > 50k', 'earns > 50k', 'earns < 50k', 'earns < 50k', 'earns < 50k']]

def render_demonstrations(examples):
    if len(examples) != 6 or len(examples[-1]) != 5:
        raise ValueError("Adult requires exactly five feature rows and five labels")
    result = []
    for i, (features, label) in enumerate(zip(examples[:5], examples[-1]), 1):
        if len(features) != len(feature_names) or label not in class_names:
            raise ValueError("Adult demonstration features/label invalid")
        row = "; ".join(f"{name}: {value}" for name, value in zip(feature_names, features))
        result.append(f"Demonstration {i}: {row}\nLabel: {label}")
    return "\n\n".join(result)

def get_llm_label(s):
    pattern = r'<answer>\s*(earns > 50k|earns < 50k)\s*</answer>'
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
    rounded_list = [(str(item[0]), round(float(item[1]), 2)) for item in data_list]
    rounded_string = str(rounded_list)
    return rounded_string

# --- Prompt construction functions ---
def make_prompt_control(row, examples):
    prompt = f"""
        Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a
        study to examine people's reliance on AI systems with or without explanations for the AI's predictions.

        The domain is Adult Census and the features in order are: {feature_names}

        The possible labels are: {class_names}

        Here are five training demonstrations, each paired with its correct label:
        {render_demonstrations(examples)}

        Category guide: Married-civ-spouse means married to a civilian spouse; Married-AF-spouse means married to someone in the armed forces. A question mark means the value was not recorded, not a new category.
        The label 'earns < 50k' is the legacy answer token for annual income at or below $50,000; 'earns > 50k' means above $50,000.
        Use only the supplied information; do not assume a person's income from demographic stereotypes.

        Your task is to look at a new test instance, and decide what you think the real label is.

        Here is the new test instance: {row.test_instance}

        Outline your reasoning process step by step, and give your final answer as one of the possible class labels in the tags <answer>...</answer>.
        """
    return prompt

def make_prompt_saliency(row, examples):
    if str(row.explanation_target_label) != str(row.test_predicted_label):
        raise ValueError("Saliency target does not match displayed prediction")
    prompt = f"""
        Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a
        study to examine people's reliance on AI systems with or without explanations for the AI's predictions.

        The domain is Adult Census and the features in order are: {feature_names}

        The possible labels are: {class_names}

        Here are five training demonstrations, each paired with its correct label:
        {render_demonstrations(examples)}

        Category guide: Married-civ-spouse means married to a civilian spouse; Married-AF-spouse means married to someone in the armed forces. A question mark means the value was not recorded, not a new category.
        The label 'earns < 50k' is the legacy answer token for annual income at or below $50,000; 'earns > 50k' means above $50,000.
        Use only the supplied information; do not assume a person's income from demographic stereotypes.

        Your task is to look at a new test instance, and an AI's prediction for its label, and decide what you think the real label is.

        Here is the new test instance: {row.test_instance}

        Here is the AI prediction for the new test instance: {row.test_predicted_label}

        In addition, here is a feature saliency explanation for the AI prediction where you see each feature, 
        and its contribution (positive or negative) towards the prediction: {round_numbers_in_string(row.lime_explanation)}    

        Outline your reasoning process step by step, and give your final answer as one of the possible class labels in the tags <answer>...</answer>.
        """
    return prompt

def make_prompt_nns(row, examples):
    prompt = f"""
        Pretend you are a participant in a user study with some technical machine learning knowledge being asked to assist in a
        study to examine people's reliance on AI systems with or without explanations for the AI's predictions.

        The domain is Adult Census and the features in order are: {feature_names}

        The possible labels are: {class_names}

        Here are five training demonstrations, each paired with its correct label:
        {render_demonstrations(examples)}

        Category guide: Married-civ-spouse means married to a civilian spouse; Married-AF-spouse means married to someone in the armed forces. A question mark means the value was not recorded, not a new category.
        The label 'earns < 50k' is the legacy answer token for annual income at or below $50,000; 'earns > 50k' means above $50,000.
        Use only the supplied information; do not assume a person's income from demographic stereotypes.

        Your task is to look at a new test instance, and an AI's prediction for its label, and decide what you think the real label is.

        Here is the new test instance: {row.test_instance}

        Here is the AI prediction for the new test instance: {row.test_predicted_label}

        In addition, here is an explanation for the prediction in the form of the two nearest neighbors from the training data
        that the AI learned from: 

        The first nearest neighbor is {row.neighbor1_instance},
        its label is '{  row.neighbor1_true_label }',
        and the AI's prediction of this nearest neighbor is '{  row.neighbor1_predicted_label }'
        The second nearest neighbor is {row.neighbor2_instance},
        its label is '{  row.neighbor2_true_label }',
        and the AI's prediction of this nearest neighbor is '{  row.neighbor2_predicted_label }'

        Outline your reasoning process step by step, and give your final answer as one of the possible class labels in the tags <answer>...</answer>.
        """
    return prompt

prompt_mapping = {
    "control": make_prompt_control,
    "saliency": make_prompt_saliency,
    "nns": make_prompt_nns
}

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

if __name__ == "__main__":
    # Configuration via argparse
    parser = argparse.ArgumentParser(description="Run AI reliance experiment with configurable model, batch size, and temperature.")
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
        
        df = pd.read_csv("data_v2/adult_neighbors_and_explanations_seed_"+str(seed)+".csv")
        df = df.replace({0: "earns < 50k", 1: "earns > 50k"})

        if limit is not None:
            df = df.head(limit).copy()
            print(f"[sanity] seed {seed}: limited to first {len(df)} test instance(s) "
                  f"-> {len(df) * len(prompt_mapping) * num_repetitions} prompts")

        # --- examples ---
        examples = ADULT_EXAMPLES
        print("Examples:")
        print(examples)

        # Initialize columns
        for key in prompt_mapping.keys():
            df[f'{key}_labels'] = None
            df[f'{key}_texts'] = None

        # Step 1: Gather all prompts
        print(f"Gathering all prompts for seed {seed}...")
        all_prompts = []
        prompt_metadata = []  # Store (row_idx, prompt_type, repetition_idx)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating prompts"):
            for key, prompt_func in prompt_mapping.items():
                for rep in range(num_repetitions): # Using the hyperparameter here
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
                {"study": "adult", "instance_id": f"seed:{seed}:row:{idx}:rep:{rep}", "condition": key,
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
        out_dir = "data/reliance/adult_results"
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_limit{limit}" if limit is not None else ""
        df.to_csv(
            os.path.join(
                out_dir,
                f"results_adult_seed_{seed}_{model_name}{suffix}.csv",
            ),
            index=False
        )
        print(f"Saved results for seed {seed}")
        
        
        
        
        
        
        
        
