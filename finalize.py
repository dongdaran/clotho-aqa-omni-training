"""Evaluate the selected ms-swift validation checkpoint on test once."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path

from swift_train import evaluate_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--best-tsv', type=Path, default=Path('results/optuna_budgeted/best.tsv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/optuna_final'))
    args = parser.parse_args()
    with args.best_tsv.open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if len(rows) != 1 or rows[0]['state'] != 'COMPLETE':
        raise ValueError('Expected one eligible best configuration')
    row = rows[0]
    identity = {
        'selection_sha256': hashlib.sha256(args.best_tsv.read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256((args.data_dir / 'test.jsonl').read_bytes()).hexdigest(),
    }
    result_path = args.output_dir / 'result.json'
    if result_path.exists():
        if json.loads(result_path.read_text()).get('identity') != identity:
            raise ValueError('Existing test result belongs to different data/selection; use a new output directory.')
        print(result_path.read_text())
        return
    adapter = Path(row['adapter'])
    if not (adapter / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Selected adapter is missing: {adapter}')
    test = evaluate_adapter(adapter, args.data_dir, args.output_dir)
    result = {'identity': identity, 'selection': row, 'test': test}
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
