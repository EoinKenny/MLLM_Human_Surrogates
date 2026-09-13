import sys
import os
# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import time
import re
import ast
import multiprocessing
import random
import hashlib
import json

from pandas import Series as pd_Series
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from llm_query import call_llm

# ========================
# CONFIGURATION (defaults; overridden by argparse)
# ========================
BATCH_SIZE = 5  # Adjust this to control batch size for API calls
MODEL_NAME = 'gpt5'  # Options: 'claude4.8-opus', 'claude4.6-sonnet', 'claude4.5-sonnet', 'gpt5.5', 'gpt5', 'gpt5-mini', 'gpt-4o', 'gpt-4o-mini', 'o4-mini'
TEMPERATURE = 1.0
NUM_TEST_INSTANCES = 300  # Number of test instances to evaluate (set to None to use all available)
NUM_RESPONSES_PER_PROMPT = 1  # Number of LLM responses to collect per prompt (for averaging/majority vote)
PREPARE_ONLY = False
RESUME_FROM_BATCH = 0  # Set to 0 to start from beginning, or specific batch number to resume

# Small epsilon for log stability
EPSILON = 1e-9

# ---------------------------
# Utility functions
# ---------------------------
def get_llm_label(s):
    """ Extract label from <answer> tags """
    pattern = r'<answer>\s*(0|1|1\.0|0\.0)\s*</answer>'
    match = re.search(pattern, s, re.IGNORECASE)
    if match:
        return int(float(match.group(1)))
    else:
        return None


def process_batch_with_retry(batch_prompts, model_name, temperature, max_retries=3):
    """
    Process a batch with retry logic for transient errors.
    
    Args:
        batch_prompts: List of prompts to process
        model_name: Name of the model to use
        temperature: Temperature parameter for the model
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of responses from the LLM
    """
    for attempt in range(max_retries):
        try:
            return call_llm(
                prompts=batch_prompts,
                model_name=model_name,
                temperature=temperature
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Exponential backoff: 30s, 60s, 90s
                print(f"\n❌ Batch failed with error: {type(e).__name__}: {e}")
                print(f"⏳ Retrying in {wait_time} seconds (attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Batch failed after {max_retries} attempts: {e}")
                raise
    return []


def save_checkpoint(output_dir, model_name, num_test_instances, num_responses, 
                   batch_idx, all_responses, prompt_metadata):
    """Save checkpoint for resuming later."""
    checkpoint_file = os.path.join(
        output_dir, 
        f"checkpoint_{model_name}_n{num_test_instances}_r{num_responses}.json"
    )
    checkpoint_data = {
        'last_completed_batch': batch_idx,
        'num_responses': len(all_responses),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"💾 Checkpoint saved: batch {batch_idx + 1} completed")


def load_checkpoint(output_dir, model_name, num_test_instances, num_responses):
    """Load checkpoint if it exists."""
    checkpoint_file = os.path.join(
        output_dir, 
        f"checkpoint_{model_name}_n{num_test_instances}_r{num_responses}.json"
    )
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"\n📂 Checkpoint found: {checkpoint_file}")
        print(f"   Last completed batch: {checkpoint['last_completed_batch'] + 1}")
        print(f"   Responses collected: {checkpoint['num_responses']}")
        print(f"   Timestamp: {checkpoint['timestamp']}")
        return checkpoint['last_completed_batch'] + 1
    return None


def load_intermediate_results(output_dir, model_name, num_test_instances, num_responses, last_batch):
    """Load intermediate results from the last saved batch."""
    temp_filename = os.path.join(
        output_dir, 
        f"experiment_{model_name}_n{num_test_instances}_r{num_responses}_intermediate_batch_{last_batch}.csv"
    )
    if os.path.exists(temp_filename):
        print(f"📥 Loading intermediate results from: {temp_filename}")
        temp_df = pd.read_csv(temp_filename)
        
        # Parse the stored metadata and responses
        all_responses = temp_df['response'].tolist()
        prompt_metadata = []
        for meta_str in temp_df['prompt_metadata']:
            # Convert string representation back to dict
            meta_dict = eval(meta_str)
            prompt_metadata.append(meta_dict)
        
        print(f"✅ Loaded {len(all_responses)} existing responses")
        return all_responses, prompt_metadata
    else:
        print(f"⚠️  Warning: Could not find intermediate file: {temp_filename}")
        return [], []

    
def run_experiment():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Seed: {seed}")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Test instances: {NUM_TEST_INSTANCES if NUM_TEST_INSTANCES else 'ALL'}")
    print(f"Responses per prompt: {NUM_RESPONSES_PER_PROMPT}")
    print(f"Resume from batch: {RESUME_FROM_BATCH if RESUME_FROM_BATCH > 0 else 'Start from beginning'}")
    print(f"{'='*60}\n")
    
    # Create output directory with model name
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # ====================================================
    # 1. DATA GENERATION
    # ====================================================
    feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5']
    binary_features = ['X1', 'X2']
    continuous_features = ['X3', 'X4', 'X5']

    def compute_Y(x1, x2, x3, x4, x5, threshold=4.71):
        log_x3 = np.log(x3 + EPSILON)
        log_x4 = np.log(x4 + EPSILON)
        score = (x1 + x2 + 1.2 * log_x3 + log_x4 + 1.5 * x5
                 + 0.1 * x1 * log_x3 + 0.1 * x2 * log_x4)
        return (score >= threshold).astype(int) if isinstance(score, np.ndarray) else int(score >= threshold)

    def generate_data(generation_seed, n_samples):
        rng = np.random.RandomState(generation_seed)
        X1 = rng.binomial(1, 0.5, n_samples)
        X2 = rng.binomial(1, 0.5, n_samples)
        X3 = np.exp(0.5 + 0.8 * X1 + 0.3 * rng.randn(n_samples)).round(2)
        X4 = np.exp(0.7 + X2 + 0.3 * rng.randn(n_samples)).round(2)
        X5 = np.exp(0.1 * np.log(X3 + EPSILON) + 0.1 * np.log(X4 + EPSILON)
                    + 0.3 * rng.randn(n_samples)).round(2)
        Y = compute_Y(X1, X2, X3, X4, X5)
        return pd.DataFrame(dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5, Y=Y))

    # Recreate the original training split solely for test/train separation.
    data = generate_data(seed, 10000)
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=data['Y'])

    # Exact five teaching pairs from the current experiment. The saved CFs use
    # nearest selected-feature flips on the 0.01 grid with fixed residuals.
    # They are not regenerated by the old median-reflection/noise procedure.
    num_total_examples = 5
    balanced_selected_counterfactuals = [{'source_row': 6773,
      'feature': 'X1',
      'original_instance': {'X1': 0.0, 'X2': 1.0, 'X3': 1.43, 'X4': 5.79, 'X5': 0.74, 'Y': 0.0},
      'cf_instance': {'X1': 1.0, 'X2': 1.0, 'X3': 3.18, 'X4': 5.79, 'X5': 0.8, 'Y': 1.0}},
     {'source_row': 5728,
      'feature': 'X2',
      'original_instance': {'X1': 0.0, 'X2': 0.0, 'X3': 2.25, 'X4': 1.71, 'X5': 1.09, 'Y': 0.0},
      'cf_instance': {'X1': 0.0, 'X2': 1.0, 'X3': 2.25, 'X4': 4.65, 'X5': 1.2, 'Y': 1.0}},
     {'source_row': 4123,
      'feature': 'X3',
      'original_instance': {'X1': 1.0, 'X2': 0.0, 'X3': 2.63, 'X4': 2.69, 'X5': 0.8, 'Y': 0.0},
      'cf_instance': {'X1': 1.0, 'X2': 0.0, 'X3': 3.17, 'X4': 2.69, 'X5': 0.82, 'Y': 1.0}},
     {'source_row': 7505,
      'feature': 'X4',
      'original_instance': {'X1': 1.0, 'X2': 1.0, 'X3': 4.06, 'X4': 7.86, 'X5': 1.52, 'Y': 1.0},
      'cf_instance': {'X1': 1.0, 'X2': 1.0, 'X3': 4.06, 'X4': 0.46, 'X5': 1.14, 'Y': 0.0}},
     {'source_row': 6040,
      'feature': 'X5',
      'original_instance': {'X1': 1.0, 'X2': 0.0, 'X3': 3.88, 'X4': 2.0, 'X5': 1.37, 'Y': 1.0},
      'cf_instance': {'X1': 1.0, 'X2': 0.0, 'X3': 3.88, 'X4': 2.0, 'X5': 0.83, 'Y': 0.0}}]

    # --- Extract Base Instances and CF pairs for prompts ---
    base_instances_for_prompts = [cf['original_instance'] for cf in balanced_selected_counterfactuals]
    print(f"Extracted {len(base_instances_for_prompts)} base instances for prompts.")
    
    control_examples_for_prompt = base_instances_for_prompts

    def extract_cf_instances_for_prompt(selected_cf_list):
        return [{'training_instance': cf['original_instance'], 'counterfactual': cf['cf_instance']}
                for cf in selected_cf_list]
    cf_examples_for_prompt = extract_cf_instances_for_prompt(balanced_selected_counterfactuals)

    causal_examples_for_prompt = []
    for cf_data in balanced_selected_counterfactuals:
        intervened_feature = cf_data['feature']
        original_instance_dict = cf_data['original_instance']
        causal_explanation_text = f"In this specific example, part of the reason the label is {original_instance_dict['Y']} is because feature {intervened_feature} was {original_instance_dict[intervened_feature]}."
        causal_examples_for_prompt.append({
            "instance": original_instance_dict,
            "causal_explanation": causal_explanation_text
        })


    # ====================================================
    # 5. PROMPT CONSTRUCTION FUNCTIONS
    # ====================================================
    def make_prompt_control(test_instance, examples):
        formatted_examples = "\n".join([f"Instance: {e}" for e in examples])
        return f"""
        Pretend you are a participant in a user study designed to (1) teach you about a complex causal domain, and (2) get you to make a prediction on a novel
        test instance given what you learned.

        The data domain has 5 features and is structured as follows:
         - X1, X2: Binary variables (0 or 1)
         - X3, X4, X5: Continuous POSITIVE variables (representing quantities > 0)
         - Y: Outcome (0 or 1)

        You are now going to learn about the underlying domain by studying the following information:
        Here are {len(examples)} training examples,
        and their labels Y to help you understand the domain:
        {formatted_examples}

        Here is the new test instance: {test_instance}

        Now, given all this information, make an informed prediction on what Y is for this test instance.
        Explain your reasoning briefly, and give your final answer as 0 or 1 in the tags <answer>...</answer>.
        """

    def make_prompt_causal(test_instance, examples):
        formatted_examples = "\n".join([f"Instance: {e['instance']}, Causal Explanation: {e['causal_explanation']}" for e in examples])
        return f"""
        Pretend you are a participant in a user study designed to (1) teach you about a complex causal domain, and (2) get you to make a prediction on a novel
        test instance given what you learned.

        The data domain has 5 features and is structured as follows:
         - X1, X2: Binary variables (0 or 1)
         - X3, X4, X5: Continuous POSITIVE variables (representing quantities > 0)
         - Y: Outcome (0 or 1)

        You are now going to learn about the underlying causal graph by studying the following explanation:
        Here are {len(examples)} training examples,
        and a causal explanation for their classification,
        and their labels Y to help you learn the underlying causal graph,
        study this carefully:
        {formatted_examples}

        Here is the new test instance: {test_instance}

        Now, given all this information, and what you have learned from the explanation, make an informed prediction on what Y is for this test instance.
        Explain your reasoning briefly, and give your final answer as 0 or 1 in the tags <answer>...</answer>.
        """

    def make_prompt_cfs(test_instance, examples):
        formatted_examples = "\n".join([f"Training Instance: {e['training_instance']}, Counterfactual Instance: {e['counterfactual']}" for e in examples])
        return f"""
        Pretend you are a participant in a user study designed to (1) teach you about a complex causal domain, and (2) get you to make a prediction on a novel
        test instance given what you learned.

        The data domain has 5 features and is structured as follows:
         - X1, X2: Binary variables (0 or 1)
         - X3, X4, X5: Continuous POSITIVE variables (representing quantities > 0)
         - Y: Outcome (0 or 1)

        You are now going to learn about the underlying causal graph by studying the following explanation:
        Here are {len(examples)} training examples,
        counterfactual versions of them where Y changes to the opposite class,
        and their labels Y to help you learn the underlying causal graph,
        study this carefully:
        {formatted_examples}

        Here is the new test instance: {test_instance}

        Now, given all this information, and what you have learned from the explanation, make an informed prediction on what Y is for this test instance.
        Explain your reasoning briefly, and give your final answer as 0 or 1 in the tags <answer>...</answer>.
        """

    # ====================================================
    # 6. PREPARE TEST DATA
    # ====================================================
    # Use the same fresh pool for simple random test sampling.
    fresh = generate_data(20260912, 20000)
    def feature_identity(row):
        return tuple(float(row[f]) for f in feature_cols)
    excluded = {feature_identity(row) for _, row in train_data.iterrows()}
    excluded.update(feature_identity(pair['cf_instance'])
                    for pair in balanced_selected_counterfactuals)
    fresh = fresh[[feature_identity(row) not in excluded for _, row in fresh.iterrows()]]

    # Simple random sampling; retain the original CLI sample-size option.
    target_n = NUM_TEST_INSTANCES if NUM_TEST_INSTANCES is not None else len(fresh)
    if not 1 <= target_n <= len(fresh):
        raise ValueError('Requested test count exceeds the available fresh pool')
    test_data = fresh.sample(n=target_n, random_state=42, replace=False)
    test_sample = test_data.copy()
    print(f"Using {len(test_data)} test instances (simple random sampling; natural class balance)\n")

    # ====================================================
    # 7. COLLECT ALL PROMPTS FOR BATCH PROCESSING
    # ====================================================
    print("Collecting all prompts for batch processing...")
    all_prompts = []
    prompt_metadata = []  # To track which prompt belongs to which instance/condition
    
    for i in tqdm(range(len(test_data)), desc="Generating Prompts"):
        test_instance_series = test_sample.iloc[i]
        test_instance_dict = test_instance_series[feature_cols].to_dict()
        test_instance_dict = {k: round(float(v), 2) if isinstance(v, (float, np.floating, np.integer)) else int(v)
                              for k, v in test_instance_dict.items()}
        ground_truth_label = int(test_instance_series['Y'])

        prompt_control = make_prompt_control(test_instance_dict, control_examples_for_prompt)
        prompt_causal = make_prompt_causal(test_instance_dict, causal_examples_for_prompt)
        prompt_cfs = make_prompt_cfs(test_instance_dict, cf_examples_for_prompt)

        # Add each prompt type NUM_RESPONSES_PER_PROMPT times
        for _ in range(NUM_RESPONSES_PER_PROMPT):
            all_prompts.append(prompt_control)
            prompt_metadata.append({'instance_idx': i, 'condition': 'control', 
                                   'test_instance': test_instance_dict, 
                                   'ground_truth': ground_truth_label})
            
            all_prompts.append(prompt_causal)
            prompt_metadata.append({'instance_idx': i, 'condition': 'causal',
                                   'test_instance': test_instance_dict,
                                   'ground_truth': ground_truth_label})
            
            all_prompts.append(prompt_cfs)
            prompt_metadata.append({'instance_idx': i, 'condition': 'cfs',
                                   'test_instance': test_instance_dict,
                                   'ground_truth': ground_truth_label})

    if PREPARE_ONLY:
        prepared = {
            'protocol': 'random300-new-causal', 'sampling_seed': 42,
            'pool_generation_seed': 20260912,
            'demonstrations': balanced_selected_counterfactuals,
            'requests': [dict(meta, prompt=prompt,
                              source_row=int(test_data.index[meta['instance_idx']]))
                         for meta, prompt in zip(prompt_metadata, all_prompts)]
        }
        with open(os.path.join(output_dir, 'teaching_prepared.json'), 'w') as f:
            json.dump(prepared, f, indent=2)
        return prepared

    total_llm_calls = len(all_prompts)
    print(f"\nTotal prompts collected: {total_llm_calls}")
    print(f"  - Test instances: {len(test_data)}")
    print(f"  - Conditions per instance: 3 (control, causal, cfs)")
    print(f"  - Responses per condition: {NUM_RESPONSES_PER_PROMPT}")
    print(f"  - Total LLM calls: {len(test_data)} × 3 × {NUM_RESPONSES_PER_PROMPT} = {total_llm_calls}")
    print(f"\nProcessing in batches of {BATCH_SIZE}...")
    
    # ====================================================
    # 8. HANDLE CHECKPOINT/RESUME
    # ====================================================
    start_batch = RESUME_FROM_BATCH
    all_responses = []
    
    if start_batch > 0:
        # Load existing responses
        checkpoint_last_batch = load_checkpoint(output_dir, MODEL_NAME, NUM_TEST_INSTANCES, NUM_RESPONSES_PER_PROMPT)
        if checkpoint_last_batch is not None and checkpoint_last_batch > start_batch:
            print(f"⚠️  Checkpoint shows batch {checkpoint_last_batch}, but RESUME_FROM_BATCH is set to {start_batch}")
            print(f"   Using RESUME_FROM_BATCH value: {start_batch}")
        
        # Try to load intermediate results
        existing_responses, existing_metadata = load_intermediate_results(
            output_dir, MODEL_NAME, NUM_TEST_INSTANCES, NUM_RESPONSES_PER_PROMPT, start_batch
        )
        
        if existing_responses:
            all_responses = existing_responses
            print(f"✅ Successfully loaded {len(all_responses)} existing responses")
            print(f"🔄 Resuming from batch {start_batch + 1}\n")
        else:
            print(f"⚠️  Could not load intermediate results. Starting fresh from batch {start_batch + 1}\n")
    
    # ====================================================
    # 9. PROCESS PROMPTS IN BATCHES WITH RETRY
    # ====================================================
    num_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(start_batch, num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(all_prompts))
        batch_prompts = all_prompts[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_idx} to {end_idx})...")
        print(f"{'='*60}")
        
        batch_start_time = time.time()
        
        # Use retry logic for batch processing
        batch_responses = process_batch_with_retry(
            batch_prompts=batch_prompts,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_retries=3
        )
        
        batch_duration = time.time() - batch_start_time
        print(f"✅ Batch completed in {batch_duration:.2f} seconds")
        
        all_responses.extend(batch_responses)
        
        # Save intermediate results after EVERY batch
        print(f"💾 Saving intermediate results after batch {batch_idx + 1}...")
        temp_df = pd.DataFrame({
            'prompt_metadata': prompt_metadata[:len(all_responses)],
            'response': all_responses
        })
        temp_filename = os.path.join(
            output_dir, 
            f"experiment_{MODEL_NAME}_n{NUM_TEST_INSTANCES}_r{NUM_RESPONSES_PER_PROMPT}_intermediate_batch_{batch_idx + 1}.csv"
        )
        temp_df.to_csv(temp_filename, index=False)
        
        # Save checkpoint
        save_checkpoint(
            output_dir, MODEL_NAME, NUM_TEST_INSTANCES, NUM_RESPONSES_PER_PROMPT,
            batch_idx, all_responses, prompt_metadata
        )
    
    # ====================================================
    # 10. ORGANIZE RESULTS BY TEST INSTANCE
    # ====================================================
    print("\n" + "="*60)
    print("ORGANIZING FINAL RESULTS")
    print("="*60)
    experiment_data = {}
    
    for metadata, response in zip(prompt_metadata, all_responses):
        instance_idx = metadata['instance_idx']
        condition = metadata['condition']
        
        if instance_idx not in experiment_data:
            experiment_data[instance_idx] = {
                'test_instance': metadata['test_instance'],
                'ground_truth_label': metadata['ground_truth'],
                'llm_labels_control': [],
                'llm_labels_causal': [],
                'llm_labels_cfs': [],
                'llm_responses_control': [],
                'llm_responses_causal': [],
                'llm_responses_cfs': []
            }
        
        label = get_llm_label(response)
        
        if condition == 'control':
            experiment_data[instance_idx]['llm_labels_control'].append(label)
            experiment_data[instance_idx]['llm_responses_control'].append(response)
        elif condition == 'causal':
            experiment_data[instance_idx]['llm_labels_causal'].append(label)
            experiment_data[instance_idx]['llm_responses_causal'].append(response)
        elif condition == 'cfs':
            experiment_data[instance_idx]['llm_labels_cfs'].append(label)
            experiment_data[instance_idx]['llm_responses_cfs'].append(response)
    
    # Convert to DataFrame
    exp_df = pd.DataFrame(list(experiment_data.values()))
    
    # Save final results
    csv_filename = os.path.join(
        output_dir, 
        f"experiment_{MODEL_NAME}_n{NUM_TEST_INSTANCES}_r{NUM_RESPONSES_PER_PROMPT}_final.csv"
    )
    exp_df.to_csv(csv_filename, index=False)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Final results saved to: {csv_filename}")
    print(f"📁 Results directory: {output_dir}")
    print(f"📊 Total test instances: {len(experiment_data)}")
    print(f"📋 Total responses collected: {len(all_responses)}")
    print(f"{'='*60}\n")
    
    return exp_df


if __name__ == "__main__":
    # Argparse to override model_name and batch_size (and optionally other knobs)
    parser = argparse.ArgumentParser(description="Run human-machine teaching experiment.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model name to use")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for LLM calls")
    parser.add_argument("--num_test_instances", type=int, default=NUM_TEST_INSTANCES, help="Number of test instances to evaluate (None for all)")
    parser.add_argument("--num_responses_per_prompt", type=int, default=NUM_RESPONSES_PER_PROMPT, help="Number of responses per prompt")
    parser.add_argument("--resume_from_batch", type=int, default=RESUME_FROM_BATCH, help="Batch number to resume from (0 to start fresh)")
    parser.add_argument("--prepare-only", action="store_true", help="Save questions and prompts without model calls")
    args = parser.parse_args()
    PREPARE_ONLY = args.prepare_only

    # Override globals with CLI args
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size
    NUM_TEST_INSTANCES = args.num_test_instances
    NUM_RESPONSES_PER_PROMPT = args.num_responses_per_prompt
    RESUME_FROM_BATCH = args.resume_from_batch

    run_experiment()
    
    
    
    