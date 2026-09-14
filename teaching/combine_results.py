"""Create the final long-format teaching CSV without discarding source responses."""
import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

CONDITIONS = ('control', 'causal', 'cfs')

def normalize(value):
    return int(float(value)) if str(value) in ('0', '1', '0.0', '1.0') else None

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(api_csv, run_dir, output):
    questions = json.loads((run_dir / 'questions.json').read_text())
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    requests = {(r['instance_id'], r['condition']): r for r in manifest['requests']}
    rows = []
    def base(model, family, i, condition, raw, label):
        q = questions[i]
        pred = normalize(label)
        return dict(schema_version='1.0', dataset_id='teaching-final-random300-r1',
                    model=model, model_family=family, question_index=i + 1,
                    question_id=q['id'], response_index=1, condition=condition,
                    **{f'X{k}': q['instance'][f'X{k}'] for k in range(1, 6)},
                    ground_truth=q['gold'], predicted_label='' if pred is None else pred,
                    answer_valid=int(pred is not None), correct=int(pred == q['gold']),
                    saved_label_json=json.dumps(label), raw_response=raw,
                    response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    response_characters=len(raw), test_sampling='simple_random_without_replacement',
                    test_selection_seed=42, test_pool_seed=20260912,
                    shortcut_consistent=int(q['shortcut_consistent']), score_margin=q['score_margin'],
                    prompt='', system_prompt='', prompt_sha256='', prompt_provenance='unavailable_in_API_export',
                    model_id='', model_revision='', generation_parameters_json='',
                    output_tokens='', input_tokens='', finish_reason='', request_seed='',
                    timestamp_utc='', source_filename='', source_sha256='', source_record_json='')
    with api_csv.open(newline='') as f:
        api = list(csv.DictReader(f))
    api_hash = sha(api_csv)
    models = list(dict.fromkeys(r['model'] for r in api))
    if len(models) != 9:
        raise ValueError('Expected nine API models')
    for model in models:
        records = [r for r in api if r['model'] == model]
        assert len(records) == len(questions) == 300
        for i, r in enumerate(records):
            assert ast.literal_eval(r['test_instance']) == questions[i]['instance']
            assert int(r['ground_truth_label']) == questions[i]['gold']
            for condition in CONDITIONS:
                responses = ast.literal_eval(r['llm_responses_' + condition])
                labels = ast.literal_eval(r['llm_labels_' + condition])
                assert len(responses) == len(labels) == 1 and isinstance(responses[0], str) and responses[0]
                row = base(model, 'API', i, condition, responses[0], labels[0])
                # Preserve condition-specific original CSV cells and all common source metadata.
                source = {k: v for k, v in r.items() if not k.startswith(('llm_labels_', 'llm_responses_'))}
                source.update(response_cell=r['llm_responses_' + condition], label_cell=r['llm_labels_' + condition])
                row.update(source_filename=api_csv.name, source_sha256=api_hash,
                           source_record_json=json.dumps(source, ensure_ascii=False))
                rows.append(row)
    for model in manifest['models']:
        path = run_dir / f'{model}-results.jsonl'
        source_hash = sha(path)
        records = [json.loads(l) for l in path.read_text().splitlines()]
        traces = [json.loads(l) for l in (run_dir / 'traces' / f'{model}.jsonl').read_text().splitlines()]
        by = {(r['instance_id'], r['condition']): r for r in records}
        tb = {(r['instance_id'], r['condition']): r for r in traces}
        assert len(by) == len(records) == len(tb) == len(traces) == 900
        for i, q in enumerate(questions):
            for condition in CONDITIONS:
                r, t = by[q['id'], condition], tb[q['id'], condition]
                assert r['raw_response'] == t['raw_response'] and r['gold'] == q['gold']
                assert t['prompt'] == requests[q['id'], condition]['prompt']
                row = base(model, 'open_weights', i, condition, r['raw_response'], r['answer'])
                row.update(prompt=t['prompt'], system_prompt=t['system_prompt'],
                           prompt_sha256=t['prompt_sha256'], prompt_provenance='recorded_request',
                           model_id=t['model_id'], model_revision=t['model_revision'],
                           generation_parameters_json=json.dumps(t['generation_parameters']),
                           request_seed=t['seed'], timestamp_utc=t['timestamp_utc'],
                           source_filename=path.name, source_sha256=source_hash,
                           source_record_json=json.dumps({'result': r, 'trace': t}, ensure_ascii=False))
                for key in ('input_tokens', 'output_tokens', 'finish_reason'):
                    row[key] = t['metadata'].get(key, '')
                rows.append(row)
    assert len(rows) == 10800
    assert len({(r['model'], r['question_id'], r['condition']) for r in rows}) == 10800
    assert set(Counter((r['model'], r['condition']) for r in rows).values()) == {300}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with output.open(newline='') as f:
        reread = list(csv.DictReader(f))
    assert len(reread) == len(rows)
    assert all(a['raw_response'] == b['raw_response'] for a, b in zip(rows, reread))
    output.with_suffix(output.suffix + '.sha256').write_text(sha(output) + '  ' + output.name + '\n')
    print(json.dumps({'rows': len(rows), 'models': len({r['model'] for r in rows}), 'sha256': sha(output)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-csv', type=Path, required=True)
    parser.add_argument('--open-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.api_csv, args.open_run, args.output)
