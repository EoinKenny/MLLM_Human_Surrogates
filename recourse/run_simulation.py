import pandas as pd
import json
import time
import numpy as np
import os
import sys
from tqdm import tqdm
import re

# ==========================================
# IMPORT SETUP
# ==========================================
# Add project root to sys.path so we can import the shared root llm_query module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm_query import call_llm

# ==========================================
# CONFIG CONSTANTS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(CURRENT_DIR, "data_v2", "german_experiment_prompts_actionable.csv")

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLES_PER_PROMPT = 1   # Fixed: number of LLM responses to collect per prompt

# Model configurations
model_ids = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "llama3.2-vision-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

# Batch size configuration
def get_batch_size(model_name):
    """Local inference is deliberately conservative: one prompt at a time."""
    return 1

# ==========================================
# ROBUST PARSING FUNCTIONS
# ==========================================
def parse_llm_response(resp_str):
    """
    Parse LLM response to extract JSON data containing acceptance_score and actionability_score.
    Handles corrupted, truncated, and malformed responses.

    Returns a dict with parsed fields, or None if parsing fails.
    """
    if not resp_str or not isinstance(resp_str, str):
        return None

    # Clean up the response string
    resp_str = resp_str.strip()

    # Method 1: Try direct JSON parsing first
    try:
        parsed_json = json.loads(resp_str)
        if any(key in parsed_json for key in ["acceptance_score", "actionability_score"]):
            return parsed_json
    except json.JSONDecodeError:
        pass

    # Method 2: Extract and try to repair JSON
    try:
        # Find JSON boundaries
        first_brace = resp_str.find("{")
        last_brace = resp_str.rfind("}")

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = resp_str[first_brace : last_brace + 1]

            # Try parsing as-is
            try:
                parsed_json = json.loads(json_candidate)
                if any(key in parsed_json for key in ["acceptance_score", "actionability_score"]):
                    return parsed_json
            except json.JSONDecodeError:
                pass

            # Try to repair common issues
            repaired_json = repair_json(json_candidate)
            if repaired_json:
                try:
                    parsed_json = json.loads(repaired_json)
                    if any(key in parsed_json for key in ["acceptance_score", "actionability_score"]):
                        return parsed_json
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass

    # Method 3: Extract scores using regex as fallback
    try:
        scores = extract_scores_with_regex(resp_str)
        if scores:
            return scores
    except Exception:
        pass

    return None


def repair_json(json_str):
    """
    Attempt to repair common JSON formatting issues.
    """
    try:
        repaired = json_str

        # Fix truncated field names (like "_score" -> "acceptance_score")
        repaired = re.sub(r'"_score"', '"acceptance_score"', repaired)

        # Fix truncated keys (action -> actionability_score)
        repaired = re.sub(r'"action":\s*}', '"actionability_score": 1}', repaired)

        # Fix missing quotes around field names
        repaired = re.sub(r'(\w+):', r'"\1":', repaired)

        # Fix trailing commas
        repaired = re.sub(r',\s*}', "}", repaired)
        repaired = re.sub(r',\s*]', "]", repaired)

        # Fix missing commas between fields (simple heuristic)
        repaired = re.sub(r'"\s*\n\s*"', '",\n    "', repaired)

        # Try to close unclosed strings
        if repaired.count('"') % 2 != 0:
            repaired += '"'

        # Try to close unclosed braces
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)

        return repaired
    except Exception:
        return None


def extract_scores_with_regex(resp_str):
    """
    Extract scores using regex patterns as a last resort.
    Returns dict with acceptance_score and/or actionability_score if found.
    """
    try:
        scores = {}

        # acceptance_score
        acceptance_match = re.search(r'"?acceptance_score"?\s*:\s*(\d+)', resp_str, re.IGNORECASE)
        if acceptance_match:
            scores["acceptance_score"] = int(acceptance_match.group(1))

        # actionability_score
        actionability_match = re.search(r'"?actionability_score"?\s*:\s*(\d+)', resp_str, re.IGNORECASE)
        if actionability_match:
            scores["actionability_score"] = int(actionability_match.group(1))

        # Alternative patterns: pick up any "*score" fields
        if not scores:
            score_matches = re.findall(r'"?(\w*score)"?\s*:\s*(\d+)', resp_str, re.IGNORECASE)
            for field, value in score_matches:
                if "accept" in field.lower():
                    scores["acceptance_score"] = int(value)
                elif "action" in field.lower():
                    scores["actionability_score"] = int(value)

        return scores if scores else None
    except Exception:
        return None


def run_model_evaluation(model_name, model_id, batch_size, df):
    """
    Run evaluation for a single model.
    """
    # Include model_name in output filename
    safe_model_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in model_name)
    OUTPUT_FILE = os.path.join(DATA_DIR, f"german_experiment_results_{safe_model_name}.csv")

    print(f"\n{'='*60}")
    print("German Credit Dataset - LLM Evaluation")
    print(f"{'='*60}")
    print(f"Model name: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Samples per prompt: {SAMPLES_PER_PROMPT}")
    print(f"Batch size: {batch_size}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    print(f"Processing {len(df)} prompts from German Credit dataset")
    expected_total = len(df) * SAMPLES_PER_PROMPT
    print(f"Will generate {expected_total} total responses ({SAMPLES_PER_PROMPT} per prompt)\n")

    all_results = []
    failed_parses = 0
    partial_parses = 0
    total_api_calls = 0

    # Collect SAMPLES_PER_PROMPT iterations
    for sample_num in range(SAMPLES_PER_PROMPT):
        print(f"\n{'='*60}")
        print(f"Collecting Sample {sample_num + 1} of {SAMPLES_PER_PROMPT}")
        print(f"{'='*60}\n")

        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc=f"Sample {sample_num + 1}"):
            # 1. Slice the batch
            batch_df = df.iloc[i : i + batch_size]
            prompts = batch_df["prompt_text"].tolist()
            batch_metadata = [
                {"instance_id": str(df.loc[row_id, "instance_id"]), "condition": "rating",
                 "sample_id": sample_num + 1}
                for row_id in batch_df.index
            ]

            try:
                # 2. Call LLM with list of prompts - use model_name, not model_id
                raw_responses = call_llm(
                    prompts, model_name=model_name, temperature=1.0,
                    trace_metadata=batch_metadata)
                total_api_calls += 1

                # 3. Parse and store results
                for row_idx, resp_str in enumerate(raw_responses):
                    row_data = batch_df.iloc[row_idx].to_dict()

                    # Add sample identifier and model_name
                    row_data["model_name"] = model_name
                    row_data["model_id"] = model_id
                    row_data["sample_id"] = sample_num + 1
                    row_data["original_row_id"] = i + row_idx
                    # Use the robust parsing function
                    llm_data = parse_llm_response(resp_str)

                    if llm_data:
                        # Check if we got both scores
                        has_acceptance = "acceptance_score" in llm_data
                        has_actionability = "actionability_score" in llm_data

                        if has_acceptance and has_actionability:
                            # Full parse success
                            row_data["acceptance_score"] = llm_data.get("acceptance_score")
                            row_data["actionability_score"] = llm_data.get("actionability_score")

                            # Add any additional keys (like reasoning)
                            for key, value in llm_data.items():
                                if key not in ["acceptance_score", "actionability_score"]:
                                    row_data[key] = value

                            all_results.append(row_data)
                        else:
                            # Partial parse - got some scores but not all
                            partial_parses += 1
                            row_data["acceptance_score"] = llm_data.get("acceptance_score")
                            row_data["actionability_score"] = llm_data.get("actionability_score")
                            all_results.append(row_data)

                            if partial_parses <= 5:  # Only show first 5
                                print(
                                    f"\nPartial parse for prompt {i + row_idx}, sample {sample_num + 1}: "
                                    f"missing {'acceptance' if not has_acceptance else 'actionability'}_score"
                                )
                    else:
                        failed_parses += 1
                        # Still add the row but with null scores for tracking
                        row_data["acceptance_score"] = None
                        row_data["actionability_score"] = None
                        all_results.append(row_data)

                        if failed_parses <= 5:  # Only show first 5 failures
                            print(f"\nFailed to parse response for prompt {i + row_idx}, sample {sample_num + 1}")
                            preview = resp_str[:200] if isinstance(resp_str, str) else str(resp_str)
                            print(f"Response preview: {preview}...")

            except Exception as e:
                print(f"\nError in batch starting at index {i}, sample {sample_num + 1}: {e}")
                continue

    # Save final results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_FILE, index=False)

        # Calculate statistics
        total_responses = len(all_results)
        successful_parses = total_responses - failed_parses - partial_parses
        success_rate = (successful_parses / total_responses) * 100 if total_responses > 0 else 0

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE FOR {model_name}!")
        print(f"{'='*60}")
        print(f"\nDataset: German Credit")
        print(f"Model: {model_name}")
        print(f"Original prompts: {len(df)}")
        print(f"Samples per prompt: {SAMPLES_PER_PROMPT}")
        print(f"Expected total responses: {expected_total}")
        print(f"Actual total responses: {total_responses}")
        print(f"\nParsing Results:")
        print(f"  Successful (both scores): {successful_parses} ({success_rate:.1f}%)")
        print(f"  Partial (one score): {partial_parses} ({(partial_parses/total_responses)*100:.1f}%)")
        print(f"  Failed (no scores): {failed_parses} ({(failed_parses/total_responses)*100:.1f}%)")
        print(f"\nTotal API calls made: {total_api_calls}")
        print(f"\nResults saved to: {OUTPUT_FILE}")

        # Show sample distribution
        if "sample_id" in results_df.columns:
            sample_counts = results_df["sample_id"].value_counts().sort_index()
            print(f"\nResponses per sample:")
            for sample_id, count in sample_counts.items():
                print(f"  Sample {sample_id}: {count} responses")

        print(f"\n{'='*60}\n")
        return True
    else:
        print(f"\nERROR: No results to save for {model_name}. Check your input data and LLM responses.")
        return False


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print("STARTING MULTI-MODEL EVALUATION")
    print(f"{'#'*60}")
    print(f"Total models to process: {len(model_ids)}")
    print(f"Output directory: {DATA_DIR}")
    print(f"{'#'*60}\n")

    # Load prompts once
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✓ Loaded {len(df)} prompts from {INPUT_FILE}\n")
    except FileNotFoundError:
        print(f"ERROR: File {INPUT_FILE} not found.")
        print("Please run the prompt generation script first to create:")
        print(f"  {INPUT_FILE}")
        sys.exit(1)

    # Track overall progress
    completed_models = []
    failed_models = []

    # Loop through all models
    for idx, (model_name, model_id) in enumerate(model_ids.items(), 1):
        print(f"\n{'#'*60}")
        print(f"PROCESSING MODEL {idx}/{len(model_ids)}: {model_name}")
        print(f"{'#'*60}\n")
        
        batch_size = get_batch_size(model_name)
        
        try:
            success = run_model_evaluation(model_name, model_id, batch_size, df)
            if success:
                completed_models.append(model_name)
            else:
                failed_models.append(model_name)
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"CRITICAL ERROR processing {model_name}: {e}")
            print(f"{'!'*60}\n")
            failed_models.append(model_name)
            continue

        # Small delay between models to avoid rate limiting
        if idx < len(model_ids):
            print(f"\nWaiting 5 seconds before next model...")
            time.sleep(5)

    # Final summary
    print(f"\n{'#'*60}")
    print("ALL MODELS PROCESSED - FINAL SUMMARY")
    print(f"{'#'*60}")
    print(f"\nTotal models: {len(model_ids)}")
    print(f"Successfully completed: {len(completed_models)}")
    print(f"Failed: {len(failed_models)}")
    
    if completed_models:
        print(f"\n✓ Completed models:")
        for model in completed_models:
            print(f"  - {model}")
    
    if failed_models:
        print(f"\n✗ Failed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    print(f"\nAll results saved in: {DATA_DIR}")
    print(f"{'#'*60}\n")
    
    
    
    
