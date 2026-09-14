# Final human–machine teaching dataset

UTF-8 CSV, one row per model × question × condition: 12 models × 300 questions × 3 conditions = 10,800 rows. Responses may contain newlines, quotation marks and commas: use a CSV parser, not line splitting. Data are distributed separately from this repository.

## Analysis fields

- `schema_version`, `dataset_id`: schema and study identifiers.
- `model`, `model_family`: source model name and API/open_weights group. API names are supplied aliases, not independently verified checkpoint identifiers.
- `question_index`: one-based shared question order. `question_id`: stable generator source identifier.
- `response_index`: 1; one returned response per model/question/condition.
- `condition`: control, causal (new local wording in the final protocol), or cfs (counterfactual).
- `X1`–`X5`: numeric test features; X1/X2 are binary, X3–X5 positive continuous.
- `ground_truth`: 0 or 1.
- `predicted_label`: normalized saved prediction, 0/1 or blank if invalid. Accepts integer and decimal 0/1 labels, without extracting replacement answers from prose.
- `answer_valid`: 1 for a valid normalized prediction, otherwise 0.
- `correct`: 1 if prediction matches ground truth; invalid answers count as 0. This is the primary accuracy score.
- `saved_label_json`: original parser output, encoded as JSON; preserves strict/decimal/null distinctions.
- `raw_response`: complete response string from the supplied source. This is returned explanation text, not a claim to access private internal reasoning. Literal escape sequences are preserved.
- `response_sha256`, `response_characters`: UTF-8 SHA-256 and Python character count of raw_response.

## Design and provenance

- `test_sampling`, `test_selection_seed`, `test_pool_seed`: frozen common sampling design. Five fixed teaching examples, 300 simple-random held-out instances, seed 42, pool seed 20260912.
- `shortcut_consistent`: whether the test label equals X1; descriptive subgroup only.
- `score_margin`: absolute distance from synthetic outcome threshold; descriptive subgroup only.
- `prompt`, `system_prompt`, `prompt_sha256`, `prompt_provenance`: recorded request information for open-weight models. API fields are blank with explicit unavailable provenance: the supplied API export does not contain actual prompts, so exact API prompt wording cannot be independently verified.
- `model_id`, `model_revision`: recorded open-weight checkpoint; blank when unavailable.
- `generation_parameters_json`, `request_seed`, `timestamp_utc`: recorded request settings; blank when unavailable. API settings are not inferred from open-weight settings.
- `input_tokens`, `output_tokens`, `finish_reason`: provider metadata where recorded, otherwise blank. Character counts are not token counts.
- `source_filename`, `source_sha256`: original input file provenance.
- `source_record_json`: lossless condition-specific source cells and metadata for API rows; full result and trace objects for open-weight rows. Includes historical path strings as provenance, not portable file locations.

All models have identical test feature values, labels and order. API full prompts, checkpoint revisions, timing, token usage and retry history are unavailable in the export. The final protocol and random questions are reproducible from run_teaching.py, but the separately maintained API backend is not included. Original source files, server scripts and completed runs remain in a private archive.

## Reproduction

```bash
python teaching/combine_results.py --api-csv INPUT.csv --open-run OPEN_RUN_DIRECTORY --output data/human_machine_teaching_final.csv
python teaching/analyze_results.py --data data/human_machine_teaching_final.csv --output outputs/teaching_analysis
```

The analysis draws 30 question indices with replacement, 1,000 times, using NumPy default_rng seed 42. Each draw is shared across all conditions and models. Boxes show the middle 50%; whiskers the middle 95%; dots show all-300 accuracy. These are sampling distributions for n=30, not confidence intervals for n=300 accuracy, new model calls, or independent participant replications. The script saves the exact sampled indices and summary statistics.
