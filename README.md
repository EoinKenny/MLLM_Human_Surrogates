# MLLM Human Surrogates

Core scripts for experiments using language models as surrogates for human participants in explanation studies. This initial source release contains no collected responses, datasets, model weights, figures, credentials, or server keep-alive scripts.

## Included experiments

- Teaching: five synthetic features, five fixed teaching pairs, 300 simple-random test questions, selection seed 42, control / local causal / counterfactual conditions.
- Appropriate reliance: Adult tabular classification and BIOS occupation classification.
- Recourse: model-verified German-credit counterfactual plans and human-readable prompts.
- Model improvement: iterative feedback in the diabetes classification task.
- System engagement: recommendation/explanation scenarios.

This is a source-only snapshot. Studies requiring external inputs cannot be reproduced until those inputs are supplied locally. Synthetic teaching examples embedded in the script are protocol constants; they are included so the prompts remain reproducible.

## Installation

Use Python 3.12 and an isolated environment. Core package versions below were read from the experiment server; optional dependencies not installed there are not pinned. This is not a complete transitive environment lock.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local GPU inference, use a separate Linux/NVIDIA environment and install `requirements-local.txt`. Hardware, drivers and inference-library versions affect numerical reproducibility. The three local models have checkpoint revisions pinned in `llm_query.py`. No model weights are distributed here. API credentials should be provided through environment variables or the provider's normal credential chain, never committed. API availability and provider support must be checked against the installed backend; this repository does not contain the separate API project's `src/llm_query.py`.

## Teaching: prepare without inference

Run commands in the indicated working directory; several original scripts resolve paths relative to it.

```bash
cd teaching
python run_teaching.py --prepare-only --num_test_instances 300
```

This writes `teaching/data/teaching_prepared.json`, containing questions, source IDs, labels, demonstrations and rendered prompts. It makes no model calls. The candidate pool uses seed 20260912 and 20,000 rows; sampling uses seed 42 after excluding feature-vector matches with training rows and teaching counterfactuals. There are no class, binary-feature or decision-boundary quotas. The threshold is 4.71. The causal sentence is:

> In this specific example, part of the reason the label is {Y} is because feature {feature} was {value}.

The five original examples have a 3/2 label split, with one X1=1/Y=0 counterexample. This breaks the perfect Y=X1 shortcut but does not eliminate all shortcuts. Counterfactual pairs are the frozen selected-feature nearest flips on the 0.01 grid; the script embeds them rather than regenerating or optimizing demonstrations based on results.

After reviewing prepared inputs, return to the repository root and explicitly enable collection against the current local snapshot:

```bash
python authorize_collection.py --approve
cd teaching
python run_teaching.py --model_name mistral-small-3.2-24b --batch_size 4
```

Repeat for `gemma3-27b-it` or `qwen3-vl-32b`. Each run collects 900 responses at the default sample size. This portable entry point preserves the user-facing runner; it is not the detached server supervisor. Its existing whole-batch retries and checkpoint logic are retained. Start a new run with fresh outputs; do not resume if the matching intermediate file is absent. Exact request seeds may differ from historical server runs because their trace metadata was different. No historical results are claimed to be reproduced bit-for-bit.

## Other experiments and required local inputs

| Study | Required input files | Commands / entry points |
|---|---|---|
| Adult / BIOS | `appropriate_reliance/datasets/adult.csv`, `train_df.csv`, `test_df.csv` | From root: `python appropriate_reliance/collect_data.py`, then `python appropriate_reliance/repair_saliency.py`. Then from `appropriate_reliance/`: `python test_adult.py --model_name MODEL` and `python test_bios.py --model_name MODEL`. |
| Recourse | `recourse/datasets/german.data` | From root: `python recourse/collect_prompts.py --output recourse/data_v2`. Then `python recourse/run_simulation.py`; inspect its model list/configuration before collection. |
| Model improvement | `model_improv/pima-indians-diabetes.csv`, `model_improv/feature_distributions.png` | From `model_improv/`: `python main.py`. Inspect model configuration first. The exact input visualization must be supplied; this release does not regenerate it. |
| System engagement | Scenarios embedded in the script | From `sys_eng/`: `python run_recommender_system.py --model_name MODEL`. |

Adult column names and BIOS input schemas are documented by the preparation code. These datasets are not interchangeable with arbitrary similarly named CSVs. Model and scenario defaults in older experiment scripts are preserved, not silently retargeted to the teaching study.

## Analysis

`analyze_run.py` is the existing multi-study analyzer. It expects an externally assembled run directory with subdirectories `knowledge_extraction`, `model_improvement`, `reliance_tabular`, `reliance_nlp`, `actionable_recommendation`, and `system_engagement`; inspect `--help` and its input paths before use. Data and historical archive-building supervisors are deliberately excluded from this minimal release. The CLI defaults to bootstrap sample size 30. Bootstrap resampling does not create independent participants, and results from different designs should not be pooled indiscriminately.

The teaching parser accepts both integer and decimal answer tags. Compare historical strict-scored results only after explicitly reconciling that scoring policy.

## Repository policy

`.gitignore` is an explicit allowlist of the reviewed source files. New files are ignored until added to the allowlist, including outputs, downloaded data, checkpoints, archives, private keys and local collection approvals. Do not use `git add -f` for generated or sensitive files. No license has been selected for this initial snapshot.

## Final teaching release

`teaching/run_teaching.py` is the single active teaching experiment entry point: five features, the fixed five demonstration pairs, 300 simple-random questions, and the new local causal sentence. Older teaching designs are retired to private archives and are not included in this repository. Other project experiments are unchanged. The exact historical server supervisor and inputs are also privately archived.

The final combined dataset covers nine API and three open-weight models in one long-format CSV, distributed privately for now. It is not committed. See [the data dictionary](teaching/DATA_DICTIONARY.md) for schema, provenance limits and scoring. `teaching/combine_results.py` assembles it from original exports; `teaching/analyze_results.py` reproduces the paired 1,000 × n=30 bootstrap plot. These tools do not make model calls.
