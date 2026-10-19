"""Validate current and legacy baseline predictions before reuse."""
import csv
import json
import math
from pathlib import Path
from experiment_utils import file_digest


def load_baseline(metrics_path: Path, val_path: Path) -> float:
    record = json.loads(metrics_path.read_text())
    if 'macro_f1' not in record:
        raise ValueError('Baseline lacks macro_f1. Rerun the baseline cell first.')
    baseline = record['macro_f1']
    if baseline is None or not 0 <= baseline <= 1:
        raise ValueError('Baseline must have both Yes and No labels and a valid macro_f1')
    current_digest = file_digest(val_path)
    previous_digest = record.get('dataset_sha256')
    if previous_digest == current_digest:
        return baseline
    if previous_digest is not None:
        raise ValueError('Baseline fingerprint differs from val.jsonl. Rerun the baseline cell first.')
    # Older baseline outputs lack the fingerprint. Prove that their predictions
    # cover exactly the current questions/labels before allowing their reuse.
    predictions_path = metrics_path.with_name(metrics_path.name.removesuffix('.metrics.json') + '.tsv')
    if not predictions_path.is_file():
        raise ValueError('Legacy baseline has no dataset fingerprint or predictions TSV. Rerun the baseline cell first.')
    from qwen_train import metrics
    current = [json.loads(line) for line in val_path.read_text().splitlines() if line.strip()]
    with predictions_path.open(newline='') as stream:
        predictions = list(csv.DictReader(stream, delimiter='\t'))
    for row in predictions:
        row['prediction'] = row['prediction'] or None
    expected = {row['id']: row['answer'] for row in current}
    actual = {row['id']: row['answer'] for row in predictions}
    if (not current or len(expected) != len(current) or len(actual) != len(predictions)
            or expected != actual):
        raise ValueError('Legacy baseline question IDs/labels differ from val.jsonl. Rerun the baseline cell first.')
    recalculated = metrics(predictions)
    for key in ('n', 'accuracy', 'macro_f1', 'invalid_outputs'):
        if (key not in record or recalculated[key] is None or
                not math.isclose(record[key], recalculated[key], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError('Legacy baseline metrics disagree with predictions. Rerun the baseline cell first.')
    print('Legacy baseline verified against current val question IDs, labels and predictions; reusing it.', flush=True)
    return baseline

