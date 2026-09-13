import re
import pickle
import numpy as np
import os
import sys
import argparse

from tqdm import tqdm

# ==========================================
# IMPORT SETUP
# ==========================================
# Add project root to sys.path so we can import 'src'
# Logic: This file is in ROOT/system_engagement/run_recommender_system.py, so we go up one level
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from llm_query import call_llm

# ---------------------------
# Define the prompt function
# ---------------------------
def make_prompt(movie_title, explanation=None):
    if explanation:
        prompt = f"""
        Pretend you are a human user of a movie recommender system.

        Imagine that you have $7 and a free evening
        coming up. You are considering going to the
        theater to see a movie, but only if there is a movie
        worth seeing. To determine if there is a movie
        worth seeing, you consult MovieLens for a
        personalized movie recommendation. MovieLens
        recommends one movie, and provides an explanation to justify the recommendation.

        The movie's title is withheld so that you judge the recommendation using the supplied information.
        For this scenario, $7 covers admission. Use only the supplied information; do not assume current ticket prices or any other history of this system.

        Here is its explanation for the recommendation: {explanation}

        The movies are rated on a scale of 1-5.
        Based on the provided explanation, please indicate on a scale of 1-7 how likely you are to see the movie.

        First, outline your reasoning step by step, and put your answer in between the tags <answer>...</answer>

        It is very important you follow this exact format.
        """
    elif explanation is None: # Explicit check for None
        prompt = f"""
        Pretend you are a human user of a movie recommender system.

        Imagine that you have $7 and a free evening
        coming up. You are considering going to the
        theater to see a movie, but only if there is a movie
        worth seeing. To determine if there is a movie
        worth seeing, you consult MovieLens for a
        personalized movie recommendation. MovieLens
        recommends one movie.

        The movie's title is withheld so that you judge the recommendation using the supplied information.
        For this scenario, $7 covers admission. Use only the supplied information; do not assume current ticket prices or any other history of this system.

        Please indicate on a scale of 1-7 how likely you are to see the movie.

        First, outline your reasoning step by step, and put your answer in between the tags <answer>...</answer>

        It is very important you follow this exact format.
        """
    else:
        raise TypeError('Wrong explanation type inserted')

    return prompt

# ---------------------------
# Utility functions for LLM interaction
# ---------------------------
def get_llm_label(s):
    """
    Use regex to find an integer between 1 and 7 inside <answer> tags.
    """
    pattern = r'<answer>\s*([1-7])\s*</answer>'
    match = re.search(pattern, s, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        print("No match found in <answer> tags.")
        print(s)
        return None


if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION via argparse
    # ==========================================
    parser = argparse.ArgumentParser(
        description="Run MovieLens recommender engagement experiment with configurable model and batch size."
    )
    parser.add_argument("--model_name", type=str, default="qwen3-vl-8b", help="Model name to use")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of prompts to process in each batch")
    parser.add_argument("--num_samples", type=int, default=6,
                        help="Number of repeat responses per explanation type")
    args = parser.parse_args()
    
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size
    TEMPERATURE = 1.0
    NUM_SAMPLES = args.num_samples   # replaces the hardcoded 2    

    
    # ---------------------------
    # Experiment setup
    # ---------------------------
    print(f"\n{'='*60}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"{'='*60}\n")

    seed = 0
    np.random.seed(seed)
    recommended_movie_title = '{omitted to avoid biasing you}'

    # Define explanation types
    explanation_keys = [
        "Explanation 1",
        "Explanation 2", 
        "Explanation 4",
        "Explanation 5",
        "Explanation 11",
        "Explanation 21"
    ]

    results = {key: [] for key in explanation_keys}

    # ---------------------------
    # Step 1: Collect all prompts
    # ---------------------------
    print("Collecting all prompts...")
    all_prompts = []
    prompt_metadata = []  # Track which prompt belongs to which sample/explanation

    count_1_2, count_3, count_4_5 = 3, 7, 23 
    avg_rating = 3.84

    for i in tqdm(range(NUM_SAMPLES), desc="Generating prompts"):
        # Define explanation texts for this sample
        explanation_texts = {
            "Explanation 1": f"Histogram of neighbors' ratings: {count_1_2} neighbors rated between 1-2, {count_3} neighbors rated 3, and {count_4_5} neighbors rated between 4-5.",
            "Explanation 2": "MovieLens has predicted correctly for you 80% of the time in the past.",
            "Explanation 4": f"Histogram of neighbors' ratings: 1 neighbor rated 1, 2 neighbors rated 2, 7 neighbors rated 3, 14 neighbors rated 4, and 9 neighbors rated 5.",
            "Explanation 5": "This movie is similar to 4 other movies that you rated 4 stars or higher.",
            "Explanation 11": None,  # No explanation
            "Explanation 21": f"Overall average rating for this movie is {avg_rating:.2f} out of 5."
        }
        
        # Generate a prompt for each explanation type
        for expl_key in explanation_keys:
            current_explanation = explanation_texts[expl_key]
            
            if expl_key == 'Explanation 11':
                prompt = make_prompt(recommended_movie_title)
            else:
                prompt = make_prompt(recommended_movie_title, current_explanation)
            
            all_prompts.append(prompt)
            prompt_metadata.append({
                'sample_idx': i,
                'explanation_key': expl_key
            })

    total_prompts = len(all_prompts)
    print(f"\nTotal prompts collected: {total_prompts}")
    print(f"  - Samples: {NUM_SAMPLES}")
    print(f"  - Explanation types per sample: {len(explanation_keys)}")
    print(f"  - Total: {NUM_SAMPLES} × {len(explanation_keys)} = {total_prompts}")
    print(f"\nProcessing in batches of {BATCH_SIZE}...")

    # ---------------------------
    # Step 2: Process prompts in batches
    # ---------------------------
    all_responses = []
    num_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(all_prompts))
        batch_prompts = all_prompts[start_idx:end_idx]
        batch_metadata = [
            {"instance_id": item["sample_idx"],
             "condition": item["explanation_key"]}
            for item in prompt_metadata[start_idx:end_idx]
        ]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (prompts {start_idx} to {end_idx})...")
        
        try:
            batch_responses = call_llm(
                prompts=batch_prompts,
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                trace_metadata=batch_metadata,
            )
            all_responses.extend(batch_responses)
        except Exception as e:
            print(f"ERROR: Batch {batch_idx + 1} failed: {e}")
            # Add None responses for failed batch
            all_responses.extend([None] * len(batch_prompts))
        
        # Save intermediate results every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Saving intermediate checkpoint...")
            output_dir = 'data/'
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_file = f'{output_dir}checkpoint_{MODEL_NAME}_batch_{batch_idx + 1}.pkl'
            checkpoint_data = {
                'responses': all_responses,
                'metadata': prompt_metadata[:len(all_responses)],
                'config': {
                    'model': MODEL_NAME,
                    'num_samples': NUM_SAMPLES,
                    'batch_size': BATCH_SIZE
                }
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

    # ---------------------------
    # Step 3: Organize results by explanation type
    # ---------------------------
    print("\nOrganizing results...")

    for metadata, response in zip(prompt_metadata, all_responses):
        expl_key = metadata['explanation_key']
        
        if response is None:
            print(f"Warning: No response for sample {metadata['sample_idx']}, {expl_key}")
            continue
        
        llm_label = get_llm_label(response)
        
        if llm_label is not None:
            results[expl_key].append(llm_label)
        else:
            print(f"Warning: Could not parse label for sample {metadata['sample_idx']}, {expl_key}")

    # ---------------------------
    # Step 4: Save final results
    # ---------------------------
    output_dir = 'data/'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}results_recommender_{MODEL_NAME}.pkl'

    # Save results with metadata
    output_data = {
        'results': results,
        'config': {
            'model': MODEL_NAME,
            'num_samples': NUM_SAMPLES,
            'batch_size': BATCH_SIZE,
            'temperature': TEMPERATURE,
            'seed': seed
        }
    }

    with open(filename, 'wb') as file:
        pickle.dump(output_data, file)

    print(f"\nFinal results saved to: {filename}")

    # Print summary statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for expl_key in explanation_keys:
        if results[expl_key]:
            mean_rating = np.mean(results[expl_key])
            std_rating = np.std(results[expl_key])
            n_responses = len(results[expl_key])
            print(f"{expl_key}: {mean_rating:.2f} ± {std_rating:.2f} (n={n_responses})")
        else:
            print(f"{expl_key}: No valid responses")
    print("="*60)

    print("\nExperiment complete!")



