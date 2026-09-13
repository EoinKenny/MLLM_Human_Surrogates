#!/usr/bin/env python
# coding: utf-8

"""
Multi-Model Diabetes Prediction Optimization Experiment
Compares Data-Centric (DCE), Model-Centric (MCE), and Hybrid (HYB) approaches
across multiple LLM models.

This version records accuracy at EVERY iteration and provides easy access to final accuracies.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import csv
import time
from typing import Dict, List, Tuple, Optional
import traceback
import os
import sys
from tqdm import tqdm
from functools import wraps

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_query import call_llm


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_IDS = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "llama3.2-vision-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

# Options: 'DCE' (Data-Centric), 'MCE' (Model-Centric), 'HYB' (Hybrid)
OPTIONS = ['DCE', 'MCE', 'HYB']

# Experiment parameters - USER CONFIGURABLE
NUM_RUNS_PER_VARIATION = 100  # Number of times to run each experiment with different random seeds
NUM_IMPROVEMENT_ITERATIONS = 5  # Number of optimization iterations per run
TEMPERATURE = 1.0

# Random Forest Configuration (UNDERFITTED for realistic baseline)
TEST_SIZE = 0.15  # 15% test, 85% train
BASE_RANDOM_STATE = 0  # Base seed, will be incremented for each run

RF_HYPERPARAMS = {
    'n_estimators': 30,
    'max_depth': 5,
    'min_samples_split': 50,
    'min_samples_leaf': 25,
    'max_features': 'sqrt',
    'max_leaf_nodes': 10,
    'random_state': 42  # Will be set dynamically during training
}

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2
MAX_RETRY_DELAY = 60
BACKOFF_MULTIPLIER = 2


# =============================================================================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# =============================================================================

def retry_with_backoff(max_retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY, 
                       max_delay=MAX_RETRY_DELAY, backoff_multiplier=BACKOFF_MULTIPLIER):
    """Decorator that retries a function with exponential backoff on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        tqdm.write(f"      ⚠ Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
                        tqdm.write(f"      ⏳ Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        tqdm.write(f"      ❌ All {max_retries} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_diabetes_data() -> pd.DataFrame:
    """Load diabetes dataset."""
    try:
        df = pd.read_csv('pima-indians-diabetes.csv')
        if 'Outcome' not in df.columns:
            df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        print("✓ Loaded Pima Indians Diabetes dataset")
    except FileNotFoundError as exc:
        raise FileNotFoundError("Required Pima source data is missing; synthetic fallback is forbidden") from exc

    return df


def calculate_baseline_accuracy(df: pd.DataFrame, random_state: int) -> float:
    """Calculate baseline accuracy for given random state."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )

    rf_params = RF_HYPERPARAMS.copy()
    rf_params['random_state'] = 42  # Keep model training consistent
    baseline_model = RandomForestClassifier(**rf_params)
    baseline_model.fit(X_train_base, y_train_base)
    baseline_accuracy = accuracy_score(y_test_base, baseline_model.predict(X_test_base))

    return baseline_accuracy


def calculate_skewness(series: pd.Series) -> str:
    """Simple skewness classification."""
    if len(series) == 0:
        return "unknown"
    
    mean_val = series.mean()
    median_val = series.median()
    std_val = series.std()
    
    if std_val == 0 or pd.isna(std_val):
        return "symmetric"
    
    threshold = 0.1 * std_val
    diff = abs(mean_val - median_val)
    
    if diff < threshold:
        return "symmetric"
    elif mean_val > median_val:
        return "right-skewed"
    else:
        return "left-skewed"


def find_outliers_iqr(series: pd.Series) -> int:
    """Count outliers using IQR method."""
    if len(series) == 0:
        return 0
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        return 0
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    return int(outliers)


def find_correlated_pairs(df: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """Find highly correlated feature pairs."""
    features = [col for col in df.columns if col != 'Outcome']
    
    if len(features) < 2:
        return []
    
    corr_matrix = df[features].corr().abs()
    
    pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if not pd.isna(corr_val) and corr_val >= threshold:
                pairs.append((features[i], features[j], corr_val))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:3]


def generate_dashboard_info(df_current: pd.DataFrame) -> Dict:
    """Generate comprehensive Data-Centric (DCE) explanations."""
    dashboard = {
        "num_samples": len(df_current),
        "num_features": len(df_current.columns) - 1,
        "class_balance": {},
        "correlated_pairs": [],
        "issue_scores": {},
        "feature_details": {},
        "overall_quality": 0
    }

    if 'Outcome' in df_current.columns:
        outcome_counts = df_current['Outcome'].value_counts()
        total = len(df_current)
        dashboard["class_balance"] = {
            "positive_pct": round(100 * outcome_counts.get(1, 0) / total, 1),
            "negative_pct": round(100 * outcome_counts.get(0, 0) / total, 1)
        }

    corr_pairs = find_correlated_pairs(df_current)
    dashboard["correlated_pairs"] = [
        {"feat1": pair[0], "feat2": pair[1], "correlation": round(pair[2], 2)}
        for pair in corr_pairs
    ]

    features = [col for col in df_current.columns if col != 'Outcome']
    total_zero_score = 0
    total_outlier_score = 0
    
    for col in features:
        series = df_current[col]
        
        zero_pct = round(100 * (series == 0).sum() / len(series), 1) if len(series) > 0 else 0
        p5 = round(series.quantile(0.05), 2) if len(series) > 0 else 0
        p95 = round(series.quantile(0.95), 2) if len(series) > 0 else 0
        outlier_count = find_outliers_iqr(series)
        outlier_pct = round(100 * outlier_count / len(series), 1) if len(series) > 0 else 0
        skew = calculate_skewness(series)
        
        dashboard["feature_details"][col] = {
            "zeros_pct": zero_pct,
            "min": round(series.min(), 2) if len(series) > 0 else 0,
            "p5": p5,
            "mean": round(series.mean(), 2) if len(series) > 0 else 0,
            "p95": p95,
            "max": round(series.max(), 2) if len(series) > 0 else 0,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "skew": skew
        }
        
        total_zero_score += zero_pct
        total_outlier_score += outlier_pct

    num_features = len(features)
    avg_zero_pct = total_zero_score / num_features if num_features > 0 else 0
    avg_outlier_pct = total_outlier_score / num_features if num_features > 0 else 0
    
    imbalance = abs(dashboard["class_balance"]["positive_pct"] - 50)
    imbalance_score = max(0, 100 - imbalance * 2)
    correlation_score = max(0, 100 - len(corr_pairs) * 10)
    
    dashboard["issue_scores"] = {
        "zeros": round(100 - avg_zero_pct, 1),
        "outliers": round(100 - avg_outlier_pct, 1),
        "imbalance": round(imbalance_score, 1),
        "correlation": round(correlation_score, 1)
    }

    dashboard["overall_quality"] = round(
        sum(dashboard["issue_scores"].values()) / len(dashboard["issue_scores"]), 1
    )

    return dashboard


def generate_model_centric_info(model, X_train: pd.DataFrame, feature_names: List[str]) -> Dict:
    """Generate Model-Centric (MCE) explanations with decision rules."""
    importances = dict(zip(feature_names, model.feature_importances_))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Get predictions from the random forest
    y_pred = model.predict(X_train)

    # Train a simple decision tree
    surrogate = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=20,
        random_state=42
    )
    surrogate.fit(X_train, y_pred)

    tree_ = surrogate.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # Extract all leaf rules
    all_rules = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [(name, "<=", threshold)])
            recurse(tree_.children_right[node], path + [(name, ">", threshold)])
        else:
            class_counts = tree_.value[node][0]
            predicted_class = int(surrogate.classes_[int(class_counts.argmax())])
            n_samples = tree_.n_node_samples[node]
            confidence = class_counts[predicted_class] / class_counts.sum()

            all_rules.append({
                'path': list(path),
                'predicted_class': predicted_class,
                'samples': n_samples,
                'confidence': confidence
            })

    recurse(0, [])

    # Split rules by class
    diabetic_rules = [r for r in all_rules if r['predicted_class'] == 1]
    non_diabetic_rules = [r for r in all_rules if r['predicted_class'] == 0]

    diabetic_rules.sort(key=lambda x: (x['samples'], x['confidence']), reverse=True)
    non_diabetic_rules.sort(key=lambda x: (x['samples'], x['confidence']), reverse=True)

    diabetic_rules = diabetic_rules[:4]
    non_diabetic_rules = non_diabetic_rules[:4]

    return {
        "all_importances": sorted_features,
        "diabetic_rules": diabetic_rules,
        "non_diabetic_rules": non_diabetic_rules
    }


def extract_json_from_response(response: str) -> str:
    """Extract JSON object from LLM response and remove comments."""
    brace_count = 0
    start_idx = None

    for i, char in enumerate(response):
        if char == '{':
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_str = response[start_idx:i+1]
                try:
                    # Remove comments before validating
                    cleaned_json = remove_json_comments(json_str)
                    json.loads(cleaned_json)
                    return cleaned_json
                except json.JSONDecodeError:
                    start_idx = None
                    continue

    # If no valid JSON found, try to clean the entire response
    cleaned = remove_json_comments(response.strip())
    return cleaned


def remove_json_comments(json_str: str) -> str:
    """Remove Python-style comments from JSON string."""
    lines = json_str.split('\n')
    cleaned_lines = []

    for line in lines:
        # Remove everything after # (comment)
        if '#' in line:
            line = line[:line.index('#')]
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def validate_and_fix_config(config_json: Dict, valid_features: List[str], 
                            feature_ranges: Dict[str, Dict]) -> Tuple[Dict, List[str]]:
    """Reject invalid settings rather than silently rewriting the model's answer."""
    if not isinstance(config_json, dict) or set(config_json) != set(valid_features):
        raise ValueError("Configuration must contain exactly the available features")
    for feat, settings in config_json.items():
        if not isinstance(settings, dict) or set(settings) != {"include", "min", "max"}:
            raise ValueError(f"Invalid settings for {feat}")
        if type(settings["include"]) is not bool:
            raise ValueError(f"include must be boolean for {feat}")
        for key in ["min", "max"]:
            x = settings[key]
            if x is not None and (type(x) not in (int, float) or not np.isfinite(x)
                or not feature_ranges[feat]["min"] <= x <= feature_ranges[feat]["max"]):
                raise ValueError(f"Invalid {key} for {feat}")
        if settings["min"] is not None and settings["max"] is not None and settings["min"] > settings["max"]:
            raise ValueError(f"Reversed bounds for {feat}")
    if not any(x["include"] for x in config_json.values()):
        raise ValueError("At least one feature must be included")
    return config_json, []


def retrain_model_with_config(X_train_orig: pd.DataFrame, y_train_orig: pd.Series,
                               X_test_orig: pd.DataFrame, y_test_orig: pd.Series,
                               config_json: Dict, 
                               original_sample_size: int,
                               valid_features: List[str],
                               feature_ranges: Dict[str, Dict]) -> Tuple[float, str, Optional[RandomForestClassifier], List[str]]:
    """Applies configuration to TRAINING data only and retrains model."""
    try:
        config_json, warnings = validate_and_fix_config(config_json, valid_features, feature_ranges)
        
        X_train = X_train_orig.copy()
        y_train = y_train_orig.copy()
        logs = []
        
        if warnings:
            logs.append(f"Config fixes: {'; '.join(warnings)}")

        # Feature Selection
        drop_cols = [f for f, s in config_json.items() 
                    if s.get("include") == False and f in X_train.columns]
        if drop_cols:
            X_train = X_train.drop(columns=drop_cols)
            logs.append(f"Dropped: {', '.join(drop_cols)}")

        # Feature Filtering - ONLY apply when values are not None
        for feat, settings in config_json.items():
            if feat in X_train.columns:
                if "min" in settings and settings["min"] is not None:
                    before_count = len(X_train)
                    mask = X_train[feat] >= settings["min"]
                    X_train = X_train[mask]
                    y_train = y_train[mask]
                    after_count = len(X_train)
                    if before_count > after_count:
                        removed = before_count - after_count
                        logs.append(f"{feat} min filter: removed {removed} rows")

                if "max" in settings and settings["max"] is not None:
                    before_count = len(X_train)
                    mask = X_train[feat] <= settings["max"]
                    X_train = X_train[mask]
                    y_train = y_train[mask]
                    after_count = len(X_train)
                    if before_count > after_count:
                        removed = before_count - after_count
                        logs.append(f"{feat} max filter: removed {removed} rows")

        if len(X_train.columns) < 1:
            error_msg = "ERROR: Too few features remaining"
            return 0.0, error_msg, None, []

        if len(X_train) < 4:
            error_msg = "ERROR: Not enough samples for training"
            return 0.0, error_msg, None, []

        # Prepare test data
        X_test = X_test_orig.copy()
        y_test = y_test_orig.copy()

        if drop_cols:
            X_test = X_test.drop(columns=[col for col in drop_cols if col in X_test.columns])

        # Train model
        rf_params = RF_HYPERPARAMS.copy()
        new_model = RandomForestClassifier(**rf_params)
        new_model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, new_model.predict(X_test))

        feature_names = list(X_train.columns)
        log_msg = "; ".join(logs) if logs else f"Kept all features, {len(X_train)} training samples"

        return accuracy, log_msg, new_model, feature_names

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        return 0.0, error_msg, None, []


def format_history_brief(iteration_history: List[Dict], max_entries: int = 10) -> str:
    """Format iteration history."""
    if not iteration_history:
        return "  No previous attempts yet (this is your first iteration)."

    recent = iteration_history[-max_entries:]
    lines = []

    for hist in recent:
        iter_num = hist['iteration']
        acc = hist['accuracy']
        change = hist['accuracy_change']
        result = "✓ IMPROVEMENT" if hist['was_best'] else "✗ NO IMPROVEMENT"

        changes = []

        if hist['features_excluded']:
            changes.append(f"excluded {', '.join(hist['features_excluded'])}")

        config = hist.get('config', {})

        filtered_features = []

        for feat, settings in config.items():
            if settings.get("include") == True:
                min_val = settings.get("min")
                max_val = settings.get("max")

                if min_val is not None or max_val is not None:
                    range_str = f"{feat}["
                    if min_val is not None:
                        range_str += f"{min_val}"
                    else:
                        range_str += "any"
                    range_str += "-"
                    if max_val is not None:
                        range_str += f"{max_val}"
                    else:
                        range_str += "any"
                    range_str += "]"
                    filtered_features.append(range_str)

        if filtered_features:
            changes.append(f"filtered: {', '.join(filtered_features)}")

        if not changes:
            changes.append("no changes (baseline)")

        change_summary = "; ".join(changes)

        lines.append(f"  - Iter {iter_num}: acc={acc:.4f} (Δ{change:+.4f}) → {result}")
        lines.append(f"    Changes: {change_summary}")

    return "\n".join(lines)


def format_current_config(config: Dict, all_features: List[str]) -> str:
    """Format current best config for display."""
    if not config:
        return "  All features included with full ranges (baseline)"

    included = []
    excluded = []
    filtered = []

    for feat in all_features:
        if feat not in config:
            included.append(feat)
        else:
            settings = config[feat]
            if settings.get("include") == False:
                excluded.append(feat)
            else:
                included.append(feat)
                if "min" in settings or "max" in settings:
                    filtered.append(f"{feat}[{settings.get('min', '?')}-{settings.get('max', '?')}]")

    lines = []
    if included:
        lines.append(f"  ✓ Included ({len(included)}): {', '.join(included)}")
    if excluded:
        lines.append(f"  ✗ Excluded ({len(excluded)}): {', '.join(excluded)}")
    if filtered:
        lines.append(f"  ↔ Filtered ranges: {', '.join(filtered)}")

    return "\n".join(lines) if lines else "  All features included"


def build_prompt(condition: str, df: pd.DataFrame, baseline_accuracy: float, 
                 current_best_accuracy: float, current_model, current_features: List[str],
                 X_train_base: pd.DataFrame, current_config: Dict, 
                 iteration_history: List[Dict], X_columns: List[str],
                 original_sample_size: int) -> str:
    """Build prompt based on condition (DCE/MCE/HYB)."""

    dce_info = generate_dashboard_info(df)
    X_train_current = X_train_base[current_features]
    mce_info = generate_model_centric_info(current_model, X_train_current, current_features)

    feature_ranges = {}
    for col in X_columns:
        feature_ranges[col] = {
            "min": round(df[col].min(), 2),
            "max": round(df[col].max(), 2)
        }

    # Add image note for DCE and HYB
    image_note = ""
    if condition in ['DCE', 'HYB']:
        image_note = "\nNOTE: An image showing feature distributions is attached for visual reference.\n"

    prompt = f"""{image_note}ROLE: You are a medical professional optimizing a diabetes prediction ML model by configuring training data.

CURRENT STATUS
- Baseline accuracy: {baseline_accuracy:.4f}
- Current best accuracy: {current_best_accuracy:.4f}
- Improvement so far: {current_best_accuracy - baseline_accuracy:+.4f}
- Original samples: {original_sample_size}, Current: {len(df)}
- Available features: {', '.join(X_columns)}

"""

    if current_config:
        prompt += f"""CURRENT BEST CONFIGURATION (what achieved {current_best_accuracy:.4f})
{format_current_config(current_config, X_columns)}

"""

    if condition in ['DCE', 'HYB']:
        prompt += f"""{'='*70}
DATA-CENTRIC SIGNALS
{'='*70}

CLASS BALANCE
- Positive class: {dce_info['class_balance']['positive_pct']:.1f}%
- Negative class: {dce_info['class_balance']['negative_pct']:.1f}%

"""
        if dce_info['correlated_pairs']:
            prompt += """HIGHLY CORRELATED PAIRS (|r| ≥ 0.8)
"""
            for pair in dce_info['correlated_pairs']:
                prompt += f"  - {pair['feat1']} ↔ {pair['feat2']}: r={pair['correlation']:.2f}\n"
            prompt += "\n"

        prompt += f"""DATA QUALITY SCORES (0-100, where 100=perfect)
  - Zeros: {dce_info['issue_scores']['zeros']:.1f}/100
  - Outliers: {dce_info['issue_scores']['outliers']:.1f}/100
  - Class balance: {dce_info['issue_scores']['imbalance']:.1f}/100
  - Correlation: {dce_info['issue_scores']['correlation']:.1f}/100
  → Overall quality: {dce_info['overall_quality']:.1f}/100

PER-FEATURE DATA SUMMARY
"""
        for feat, details in dce_info['feature_details'].items():
            issues = []
            if details['zeros_pct'] > 30:
                issues.append(f"{details['zeros_pct']:.0f}% zeros!")
            if details['outlier_pct'] > 10:
                issues.append(f"{details['outlier_pct']:.0f}% outliers!")

            issue_str = f" ⚠ {', '.join(issues)}" if issues else ""

            prompt += f"""  {feat}: [{details['min']}-{details['max']}] mean={details['mean']:.1f}, p5={details['p5']}, p95={details['p95']}
    {details['skew']}, {details['zeros_pct']:.1f}% zeros, {details['outlier_count']} outliers{issue_str}
"""

    if condition in ['MCE', 'HYB']:
        prompt += f"""{'='*70}
MODEL-CENTRIC SIGNALS
{'='*70}

FEATURE IMPORTANCE RANKING
"""
        for rank, (feat, imp) in enumerate(mce_info['all_importances'], 1):
            prompt += f"  {rank}. {feat}: {imp:.3f}\n"

        prompt += f"""
TOP DECISION RULES FOR DIABETIC (Class = 1)
"""
        for i, rule in enumerate(mce_info['diabetic_rules'], 1):
            conditions = []
            for feat, op, threshold in rule['path']:
                conditions.append(f"{feat} {op} {threshold:.2f}")
            rule_str = " AND ".join(conditions)
            prompt += f"  {i}. IF {rule_str}\n"
            prompt += f"     → Diabetic (confidence: {rule['confidence']:.1%}, samples: {rule['samples']})\n\n"

        prompt += f"""TOP DECISION RULES FOR NON-DIABETIC (Class = 0)
"""
        for i, rule in enumerate(mce_info['non_diabetic_rules'], 1):
            conditions = []
            for feat, op, threshold in rule['path']:
                conditions.append(f"{feat} {op} {threshold:.2f}")
            rule_str = " AND ".join(conditions)
            prompt += f"  {i}. IF {rule_str}\n"
            prompt += f"     → Non-Diabetic (confidence: {rule['confidence']:.1%}, samples: {rule['samples']})\n\n"

    prompt += f"""{'='*70}
OBSERVED FEATURE RANGES
{'='*70}
"""
    for feat, ranges in feature_ranges.items():
        prompt += f"  {feat}: [{ranges['min']}, {ranges['max']}]\n"

    prompt += f"""
{'='*70}
HISTORY
{'='*70}
{format_history_brief(iteration_history)}

{'='*70}
TASK: Propose a NEW configuration to beat {current_best_accuracy:.4f}
{'='*70}

OUTPUT FORMAT: 
First, provide your reasoning in a REASONING section.
Then, provide your JSON configuration.

REASONING:
<Your analysis and strategy>

CONFIGURATION:
{{
    "FeatureName": {{"include": true/false, "min": number_or_null, "max": number_or_null}}
}}

This config filters training rows and selects input columns; it does not change any patient values or labels.
Use only the supplied data. Observed ranges are dataset bounds, not recommended clinical values.
Zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI denote missing measurements; zero pregnancies is valid.

Schema requirements:
- "include": keep or exclude feature
- "min": minimum value or null
- "max": maximum value or null
- Include every listed feature exactly once. Bounds must lie within its observed range, with min <= max.
- At least one feature must remain included.

Your response:
"""

    return prompt


@retry_with_backoff(max_retries=MAX_RETRIES)
def call_llm_with_retry(prompt: str, model_name: str, temperature: float, 
                        use_image: bool = False,
                        trace_metadata: Optional[Dict] = None) -> str:
    """Call LLM API with automatic retry on failure."""
    if use_image:
        responses = call_llm([prompt], model_name, temperature, 
                           image_path="feature_distributions.png",
                           trace_metadata=trace_metadata)
    else:
        responses = call_llm([prompt], model_name, temperature,
                             trace_metadata=trace_metadata)
    
    if not responses or len(responses) == 0:
        raise ValueError("LLM returned empty response")
    return responses[0]


# =============================================================================
# OPTIMIZATION FUNCTION
# =============================================================================

def run_single_optimization(df: pd.DataFrame, baseline_accuracy: float, 
                            model_name: str, condition: str,
                            X_test_holdout: pd.DataFrame, y_test_holdout: pd.Series,
                            X_train_base: pd.DataFrame, y_train_base: pd.Series,
                            run_number: int, random_state: int,
                            pbar_iter: Optional[tqdm] = None) -> Dict:
    """Run a single optimization experiment with robust error handling.
    
    Returns:
        Dict containing:
            - summary stats (for backwards compatibility)
            - iteration_details: list of dicts, one per iteration
    """
    
    X = df.drop('Outcome', axis=1)
    
    rf_params = RF_HYPERPARAMS.copy()
    baseline_model = RandomForestClassifier(**rf_params)
    baseline_model.fit(X_train_base, y_train_base)

    current_best_accuracy = baseline_accuracy
    current_model = baseline_model
    current_config = {}
    current_features = list(X.columns)
    iteration_history = []
    iteration_details = []  # NEW: Store detailed info for each iteration
    total_api_retries = 0
    original_sample_size = len(df)
    
    feature_ranges = {}
    for col in X.columns:
        feature_ranges[col] = {
            "min": round(df[col].min(), 2),
            "max": round(df[col].max(), 2)
        }

    for iteration in range(1, NUM_IMPROVEMENT_ITERATIONS + 1):
        if pbar_iter:
            pbar_iter.set_description(f"    Iter {iteration}/{NUM_IMPROVEMENT_ITERATIONS}")
        
        iteration_success = False
        iteration_attempts = 0
        
        while not iteration_success and iteration_attempts < MAX_RETRIES:
            try:
                prompt = build_prompt(
                    condition, df, baseline_accuracy, current_best_accuracy,
                    current_model, current_features, X_train_base, current_config,
                    iteration_history, list(X.columns), original_sample_size
                )

                use_image = condition in ['DCE', 'HYB']
                llm_response_raw = call_llm_with_retry(
                    prompt, model_name, TEMPERATURE, use_image,
                    trace_metadata={
                        "instance_id": f"seed:{random_state}:iteration:{iteration}",
                        "condition": condition,
                        "run_number": run_number,
                        "random_state": random_state,
                        "iteration": iteration,
                        "attempt": iteration_attempts,
                    })

                llm_response_clean = extract_json_from_response(llm_response_raw)
                
                try:
                    new_config = json.loads(llm_response_clean)
                except json.JSONDecodeError as json_err:
                    raise ValueError(f"Invalid JSON: {str(json_err)[:100]}")

                new_accuracy, feedback, new_model_obj, new_features = retrain_model_with_config(
                    X_train_base, y_train_base, X_test_holdout, y_test_holdout,
                    new_config, original_sample_size, list(X.columns), feature_ranges
                )
                
                if new_model_obj is None or new_accuracy == 0.0:
                    raise ValueError(f"Model retraining failed: {feedback}")
                
                accuracy_change = new_accuracy - current_best_accuracy
                is_best = new_accuracy > current_best_accuracy

                # Store in iteration_history (for prompt context)
                hist_entry = {
                    'iteration': iteration,
                    'accuracy': new_accuracy,
                    'accuracy_change': accuracy_change,
                    'features_excluded': [f for f, s in new_config.items() if s.get("include") == False],
                    'features_included': [f for f, s in new_config.items() if s.get("include") == True],
                    'was_best': is_best,
                    'config': new_config,
                    'feedback': feedback
                }
                iteration_history.append(hist_entry)

                # NEW: Store detailed iteration info for CSV
                features_excluded = [f for f, s in new_config.items() if s.get("include") == False]
                features_included = [f for f, s in new_config.items() if s.get("include") == True]
                
                iteration_detail = {
                    'model_name': model_name,
                    'option': condition,
                    'run_number': run_number,
                    'random_state': random_state,
                    'iteration': iteration,
                    'accuracy': new_accuracy,
                    'accuracy_change': accuracy_change,
                    'baseline_accuracy': baseline_accuracy,
                    'improvement_from_baseline': new_accuracy - baseline_accuracy,
                    'is_best_so_far': is_best,
                    'is_final_iteration': (iteration == NUM_IMPROVEMENT_ITERATIONS),
                    'features_excluded': ','.join(features_excluded) if features_excluded else '',
                    'features_included': ','.join(features_included) if features_included else 'all',
                    'num_features_used': len(features_included) if features_included else len(X.columns),
                    'feedback': feedback[:200] if feedback else ''  # Truncate for CSV
                }
                iteration_details.append(iteration_detail)

                # Update best if needed
                if is_best:
                    current_best_accuracy = new_accuracy
                    current_config = new_config
                    current_model = new_model_obj
                    current_features = new_features

                iteration_success = True
                
                if pbar_iter:
                    pbar_iter.update(1)

            except Exception as e:
                iteration_attempts += 1
                total_api_retries += 1
                
                if iteration_attempts >= MAX_RETRIES:
                    tqdm.write(f"      ❌ Iteration {iteration} failed after {MAX_RETRIES} attempts")
                    
                    # Store failed iteration in history
                    hist_entry = {
                        'iteration': iteration,
                        'accuracy': current_best_accuracy,
                        'accuracy_change': 0.0,
                        'features_excluded': [],
                        'features_included': [],
                        'was_best': False,
                        'config': {},
                        'feedback': f"FAILED after {MAX_RETRIES} attempts: {str(e)[:100]}"
                    }
                    iteration_history.append(hist_entry)
                    
                    # NEW: Store failed iteration details
                    iteration_detail = {
                        'model_name': model_name,
                        'option': condition,
                        'run_number': run_number,
                        'random_state': random_state,
                        'iteration': iteration,
                        'accuracy': current_best_accuracy,  # Use current best
                        'accuracy_change': 0.0,
                        'baseline_accuracy': baseline_accuracy,
                        'improvement_from_baseline': current_best_accuracy - baseline_accuracy,
                        'is_best_so_far': False,
                        'is_final_iteration': (iteration == NUM_IMPROVEMENT_ITERATIONS),
                        'features_excluded': '',
                        'features_included': '',
                        'num_features_used': 0,
                        'feedback': f"FAILED: {str(e)[:150]}"
                    }
                    iteration_details.append(iteration_detail)
                    
                    if pbar_iter:
                        pbar_iter.update(1)
                    break
                else:
                    delay = INITIAL_RETRY_DELAY * (BACKOFF_MULTIPLIER ** (iteration_attempts - 1))
                    delay = min(delay, MAX_RETRY_DELAY)
                    time.sleep(delay)

    return {
        'baseline_accuracy': baseline_accuracy,
        'final_accuracy': current_best_accuracy,
        'improvement': current_best_accuracy - baseline_accuracy,
        'best_config': current_config,
        'iteration_history': iteration_history,
        'iteration_details': iteration_details,  # NEW: Per-iteration details
        'total_api_retries': total_api_retries,
        'successful_iterations': len([h for h in iteration_history if 'FAILED' not in h.get('feedback', '')])
    }


# =============================================================================
# SAVE FUNCTIONS
# =============================================================================

def save_iteration_details_to_csv(all_iteration_details: List[Dict], filename: str = None):
    """Save every single iteration's details - one row per iteration."""
    if not all_iteration_details:
        print("No iteration data to save!")
        return

    if filename is None:
        filename = f"experiment_iterations_all.csv"

    fieldnames = ['model_name', 'option', 'run_number', 'random_state', 
                  'iteration', 'accuracy', 'accuracy_change', 
                  'baseline_accuracy', 'improvement_from_baseline',
                  'is_best_so_far', 'is_final_iteration',
                  'features_excluded', 'features_included',
                  'num_features_used', 'feedback']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_iteration_details)

    print(f"✓ All iteration details saved to: {filename}")


def save_final_accuracies_to_csv(all_iteration_details: List[Dict], filename: str = None):
    """Extract and save only final accuracies for quick analysis."""
    if not all_iteration_details:
        print("No data to extract final accuracies!")
        return

    if filename is None:
        filename = f"experiment_final_accuracies.csv"

    # Filter for final iterations only
    final_data = [row for row in all_iteration_details if row['is_final_iteration']]

    fieldnames = ['model_name', 'option', 'run_number', 'random_state',
                  'baseline_accuracy', 'final_accuracy', 'improvement_from_baseline',
                  'num_features_used']
    
    final_rows = []
    for row in final_data:
        final_rows.append({
            'model_name': row['model_name'],
            'option': row['option'],
            'run_number': row['run_number'],
            'random_state': row['random_state'],
            'baseline_accuracy': row['baseline_accuracy'],
            'final_accuracy': row['accuracy'],
            'improvement_from_baseline': row['improvement_from_baseline'],
            'num_features_used': row['num_features_used']
        })
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"✓ Final accuracies saved to: {filename}")


def save_results_to_csv(results: List[Dict], filename: str = None):
    """Save experiment summary results to CSV."""
    if not results:
        print("No results to save!")
        return

    if filename is None:
        filename = f"experiment_summary.csv"

    fieldnames = results[0].keys()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Summary results saved to: {filename}")


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_full_experiment() -> Tuple[List[Dict], List[Dict]]:
    """Run the complete experiment across all models and options."""
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    df = load_diabetes_data()
    print(f"Dataset: {len(df)} samples, {len(df.columns)-1} features")
    print(f"Using Random Forest with underfitting config (test_size={TEST_SIZE})")
    print("="*80 + "\n")

    results = []
    all_iteration_details = []  # NEW: Store ALL iterations from ALL runs
    
    # Track statistics per model-option combination
    stats_tracker = {
        f"{model_name}-{option}": {
            'accuracies': [],
            'improvements': [],
            'baselines': [],
            'api_retries': 0,
            'successful_iterations': 0,
            'total_possible_iterations': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'start_time': None
        }
        for model_name in MODEL_IDS.keys()
        for option in OPTIONS
    }

    with tqdm(total=NUM_RUNS_PER_VARIATION, desc="Overall Progress (Seeds)", position=0, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_seeds:
        
        # OUTER LOOP: Seeds
        for run in range(1, NUM_RUNS_PER_VARIATION + 1):
            run_random_state = BASE_RANDOM_STATE + run
            pbar_seeds.set_description(f"Seed {run_random_state} ({run}/{NUM_RUNS_PER_VARIATION})")
            
            # Create train/test split for this seed (shared across all models/options)
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
            X_train_base, X_test_holdout, y_train_base, y_test_holdout = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=run_random_state
            )
            
            # Calculate baseline for this split
            baseline_accuracy = calculate_baseline_accuracy(df, run_random_state)
            
            with tqdm(total=len(MODEL_IDS) * len(OPTIONS), desc=f"  Seed {run_random_state} Experiments", 
                     position=1, leave=False,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar_experiments:
                
                # MIDDLE LOOP: Models
                for model_name, model_id in MODEL_IDS.items():
                    # INNER LOOP: Options
                    for option in OPTIONS:
                        experiment_name = f"{model_name}-{option}"
                        
                        # Initialize timing on first run
                        if stats_tracker[experiment_name]['start_time'] is None:
                            stats_tracker[experiment_name]['start_time'] = time.time()
                        
                        pbar_experiments.set_description(f"  Seed {run_random_state}: {experiment_name}")
                        
                        with tqdm(total=NUM_IMPROVEMENT_ITERATIONS, desc=f"    {experiment_name}", 
                                 position=2, leave=False,
                                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar_iter:
                            
                            try:
                                result = run_single_optimization(
                                    df, baseline_accuracy, model_name, option,
                                    X_test_holdout, y_test_holdout,
                                    X_train_base, y_train_base,
                                    run, run_random_state,  # NEW: Pass run info
                                    pbar_iter
                                )
                                
                                final_accuracy = result['final_accuracy']
                                improvement = result['improvement']
                                
                                # Update stats tracker
                                stats_tracker[experiment_name]['accuracies'].append(final_accuracy)
                                stats_tracker[experiment_name]['improvements'].append(improvement)
                                stats_tracker[experiment_name]['baselines'].append(baseline_accuracy)
                                stats_tracker[experiment_name]['successful_runs'] += 1
                                stats_tracker[experiment_name]['api_retries'] += result.get('total_api_retries', 0)
                                stats_tracker[experiment_name]['successful_iterations'] += result.get('successful_iterations', 0)
                                stats_tracker[experiment_name]['total_possible_iterations'] += NUM_IMPROVEMENT_ITERATIONS

                                # NEW: Add all iteration details from this run
                                all_iteration_details.extend(result['iteration_details'])

                            except Exception as e:
                                stats_tracker[experiment_name]['failed_runs'] += 1
                                stats_tracker[experiment_name]['total_possible_iterations'] += NUM_IMPROVEMENT_ITERATIONS
                                tqdm.write(f"      ❌ {experiment_name} seed {run_random_state} failed: {str(e)[:100]}")
                                traceback.print_exc()
                        
                        pbar_experiments.update(1)
            
            # SAVE INCREMENTALLY after completing each seed
            save_iteration_details_to_csv(all_iteration_details, filename="experiment_iterations_all.csv")
            save_final_accuracies_to_csv(all_iteration_details, filename="experiment_final_accuracies.csv")
            
            tqdm.write(f"\n✓ Saved results after completing seed {run_random_state} ({run}/{NUM_RUNS_PER_VARIATION})")
            tqdm.write(f"  Total iterations saved: {len(all_iteration_details)}")
            tqdm.write(f"  Total runs completed: {len([d for d in all_iteration_details if d['is_final_iteration']])}\n")
            
            pbar_seeds.update(1)

    # Compile final summary results from stats_tracker
    for model_name, model_id in MODEL_IDS.items():
        for option in OPTIONS:
            experiment_name = f"{model_name}-{option}"
            stats = stats_tracker[experiment_name]
            
            elapsed_time = time.time() - stats['start_time'] if stats['start_time'] else 0
            
            if stats['accuracies']:
                avg_accuracy = sum(stats['accuracies']) / len(stats['accuracies'])
                min_accuracy = min(stats['accuracies'])
                max_accuracy = max(stats['accuracies'])
                max_improvement = max(stats['improvements'])
                avg_improvement = sum(stats['improvements']) / len(stats['improvements'])
                avg_baseline = sum(stats['baselines']) / len(stats['baselines'])
                min_baseline = min(stats['baselines'])
                max_baseline = max(stats['baselines'])
                max_accuracy_run = stats['accuracies'].index(max_accuracy) + 1
            else:
                avg_accuracy = 0.0
                min_accuracy = 0.0
                max_accuracy = 0.0
                max_improvement = 0.0
                avg_improvement = 0.0
                avg_baseline = 0.0
                min_baseline = 0.0
                max_baseline = 0.0
                max_accuracy_run = 0

            result_entry = {
                'model_name': model_name,
                'model_id': model_id,
                'option': option,
                'avg_baseline_accuracy': avg_baseline,
                'min_baseline_accuracy': min_baseline,
                'max_baseline_accuracy': max_baseline,
                'max_accuracy': max_accuracy,
                'avg_accuracy': avg_accuracy,
                'min_accuracy': min_accuracy,
                'max_improvement': max_improvement,
                'avg_improvement': avg_improvement,
                'max_accuracy_run': max_accuracy_run,
                'total_runs': NUM_RUNS_PER_VARIATION,
                'successful_runs': stats['successful_runs'],
                'failed_runs': stats['failed_runs'],
                'improvement_iterations': NUM_IMPROVEMENT_ITERATIONS,
                'total_api_retries': stats['api_retries'],
                'successful_iterations': stats['successful_iterations'],
                'total_possible_iterations': stats['total_possible_iterations'],
                'iteration_success_rate': f"{(stats['successful_iterations']/stats['total_possible_iterations']*100):.1f}%" if stats['total_possible_iterations'] > 0 else "0%",
                'elapsed_time_seconds': elapsed_time,
            }
            results.append(result_entry)

            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"COMPLETED: {model_name} - {option}")
            tqdm.write(f"{'='*80}")
            tqdm.write(f"Avg Baseline:        {avg_baseline:.4f} (range: {min_baseline:.4f}-{max_baseline:.4f})")
            tqdm.write(f"Max Accuracy:        {max_accuracy:.4f} (run {max_accuracy_run})")
            tqdm.write(f"Avg Accuracy:        {avg_accuracy:.4f}")
            tqdm.write(f"Max Improvement:     +{max_improvement:.4f}")
            tqdm.write(f"Avg Improvement:     +{avg_improvement:.4f}")
            tqdm.write(f"Successful Runs:     {stats['successful_runs']}/{NUM_RUNS_PER_VARIATION}")
            tqdm.write(f"API Retries:         {stats['api_retries']}")
            tqdm.write(f"Iteration Success:   {stats['successful_iterations']}/{stats['total_possible_iterations']} ({result_entry['iteration_success_rate']})")
            tqdm.write(f"Time:                {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
            tqdm.write(f"{'='*80}\n")

    return results, all_iteration_details


# =============================================================================
# SUMMARY PRINTING
# =============================================================================

def print_summary(results: List[Dict]):
    """Print comprehensive summary of all results."""
    print(f"\n{'='*80}")
    print("FINAL EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    for option in OPTIONS:
        option_name = {
            'DCE': 'Data-Centric',
            'MCE': 'Model-Centric',
            'HYB': 'Hybrid (Data + Model)'
        }[option]
        
        print(f"\n{option} ({option_name}) Results:")
        print(f"{'-'*80}")
        option_results = [r for r in results if r['option'] == option]
        option_results.sort(key=lambda x: x['max_accuracy'], reverse=True)

        for i, result in enumerate(option_results, 1):
            print(f"{i:2d}. {result['model_name']:20s} | "
                  f"Max: {result['max_accuracy']:.4f} | "
                  f"Avg: {result['avg_accuracy']:.4f} | "
                  f"Max Δ: +{result['max_improvement']:.4f} | "
                  f"Avg Δ: +{result['avg_improvement']:.4f} | "
                  f"Success: {result['iteration_success_rate']}")

    print(f"\n{'='*80}")
    print("OVERALL BEST PERFORMERS")
    print(f"{'='*80}")

    best_max = max(results, key=lambda x: x['max_accuracy'])
    print(f"\n🏆 Highest Max Accuracy:")
    print(f"   Model: {best_max['model_name']} | Option: {best_max['option']}")
    print(f"   Accuracy: {best_max['max_accuracy']:.4f} | Improvement: +{best_max['max_improvement']:.4f}")

    best_avg = max(results, key=lambda x: x['avg_accuracy'])
    print(f"\n🏆 Highest Average Accuracy:")
    print(f"   Model: {best_avg['model_name']} | Option: {best_avg['option']}")
    print(f"   Accuracy: {best_avg['avg_accuracy']:.4f} | Improvement: +{best_avg['avg_improvement']:.4f}")

    best_improvement = max(results, key=lambda x: x['max_improvement'])
    print(f"\n🏆 Best Max Improvement:")
    print(f"   Model: {best_improvement['model_name']} | Option: {best_improvement['option']}")
    print(f"   Max Improvement: +{best_improvement['max_improvement']:.4f}")

    best_avg_improvement = max(results, key=lambda x: x['avg_improvement'])
    print(f"\n🏆 Best Average Improvement:")
    print(f"   Model: {best_avg_improvement['model_name']} | Option: {best_avg_improvement['option']}")
    print(f"   Avg Improvement: +{best_avg_improvement['avg_improvement']:.4f}")

    print(f"\n{'='*80}")
    print("SUMMARY BY MODEL")
    print(f"{'='*80}")

    for model_name in MODEL_IDS.keys():
        model_results = [r for r in results if r['model_name'] == model_name]
        if model_results:
            best_for_model = max(model_results, key=lambda x: x['max_accuracy'])
            avg_max_acc = sum(r['max_accuracy'] for r in model_results) / len(model_results)
            avg_improvement = sum(r['avg_improvement'] for r in model_results) / len(model_results)
            total_retries = sum(r['total_api_retries'] for r in model_results)
            
            print(f"\n{model_name}:")
            print(f"  Best: {best_for_model['max_accuracy']:.4f} ({best_for_model['option']})")
            print(f"  Avg max across options: {avg_max_acc:.4f}")
            print(f"  Avg improvement across options: +{avg_improvement:.4f}")
            print(f"  Total API retries: {total_retries}")

    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    
    total_api_retries = sum(r['total_api_retries'] for r in results)
    total_successful_runs = sum(r['successful_runs'] for r in results)
    total_runs = sum(r['total_runs'] for r in results)
    total_successful_iterations = sum(r['successful_iterations'] for r in results)
    total_possible_iterations = sum(r['total_possible_iterations'] for r in results)
    
    print(f"\nTotal Runs:              {total_runs}")
    print(f"Successful Runs:         {total_successful_runs} ({total_successful_runs/total_runs*100:.1f}%)")
    print(f"Total Iterations:        {total_possible_iterations}")
    print(f"Successful Iterations:   {total_successful_iterations} ({total_successful_iterations/total_possible_iterations*100:.1f}%)")
    print(f"Total API Retries:       {total_api_retries}")
    print(f"Avg Retries per Result:  {total_api_retries/len(results):.1f}")

    print(f"\n{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("MULTI-MODEL DIABETES PREDICTION OPTIMIZATION EXPERIMENT")
    print("="*80)
    print(f"Configuration:")
    print(f"  Models to test: {len(MODEL_IDS)}")
    print(f"  Options: {', '.join(OPTIONS)}")
    print(f"  Runs per variation: {NUM_RUNS_PER_VARIATION}")
    print(f"  Improvement iterations: {NUM_IMPROVEMENT_ITERATIONS}")
    print(f"  Max retries per API call: {MAX_RETRIES}")
    print(f"  Base random state: {BASE_RANDOM_STATE}")
    print(f"  Test size: {TEST_SIZE}")
    print(f"  RF hyperparameters: {RF_HYPERPARAMS}")
    print(f"  Total runs: {len(MODEL_IDS) * len(OPTIONS) * NUM_RUNS_PER_VARIATION}")
    print(f"  Total iterations: {len(MODEL_IDS) * len(OPTIONS) * NUM_RUNS_PER_VARIATION * NUM_IMPROVEMENT_ITERATIONS}")
    print("="*80 + "\n")

    start_time = time.time()
    results, all_iteration_details = run_full_experiment()
    total_time = time.time() - start_time

    # Save all three types of output
    save_results_to_csv(results, "experiment_summary.csv")
    save_iteration_details_to_csv(all_iteration_details, "experiment_iterations_all.csv")
    save_final_accuracies_to_csv(all_iteration_details, "experiment_final_accuracies.csv")

    print_summary(results)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes, {total_time/3600:.1f} hours)")
    print(f"\nThree output files created:")
    print(f"  1. experiment_summary.csv - High-level summary by model/option")
    print(f"  2. experiment_iterations_all.csv - Every iteration (all data)")
    print(f"  3. experiment_final_accuracies.csv - Only final accuracies (quick access)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
