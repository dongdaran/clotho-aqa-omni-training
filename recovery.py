"""Atomic prediction journals and complete training checkpoint discovery."""
import hashlib
import json
from pathlib import Path
from experiment_utils import write_json


def prediction_identity(rows, model_key):
    return hashlib.sha256(json.dumps({'rows': rows, 'model': model_key}, sort_keys=True).encode()).hexdigest()


def load_predictions(path, identity, rows):
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    results = payload['results']
    if payload['identity'] != identity or len(results) > len(rows):
        raise ValueError('Prediction journal belongs to different data/model')
    for expected, saved in zip(rows, results):
        if (expected['id'], expected['answer']) != (saved['id'], saved['answer']):
            raise ValueError('Prediction journal order/labels mismatch')
    return results


def save_predictions(path, identity, results):
    write_json(path, {'identity': identity, 'results': results})


def latest_checkpoint(root):
    valid = []
    for path in Path(root).glob('checkpoint-*'):
        if not path.name.removeprefix('checkpoint-').isdigit():
            continue
        required = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'aqa_state.json')
        if all((path / name).is_file() for name in required) and list(path.glob('rng_state*.pth')):
            valid.append(path)
    return max(valid, key=lambda p: int(p.name.split('-')[-1]), default=None)
