"""Train the selected hyperparameters from the base model for two complete augmented-data epochs."""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from experiment_utils import read_rows, source_groups, write_rows, write_json
from tune import execute_job, environment_for_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--search-dir', type=Path, default=Path('results/a100_macro_f1/search_greedy'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/a100_macro_f1/search_greedy/train_full_2epochs'))
    args = parser.parse_args()
    with (args.search_dir / 'best.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if len(rows) != 1 or rows[0]['state'] != 'COMPLETE':
        raise ValueError('Expected one completed selected configuration')
    best = rows[0]
    settings = json.loads((args.search_dir / 'search_config.json').read_text())
    runtime = json.loads((args.search_dir / 'performance.json').read_text())['selected']
    all_rows = read_rows(args.data_dir / 'train.jsonl', allow_repeated=True)
    groups = source_groups(all_rows)
    rng = random.Random(settings['seed'])
    complete_epochs = []
    for _ in range(2):
        epoch_rows = list(all_rows)
        rng.shuffle(epoch_rows)
        complete_epochs.extend(epoch_rows)
    steps = math.ceil(len(complete_epochs) / 64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / 'train_all_7_variants_2epochs.jsonl'
    write_rows(train_path, complete_epochs)
    job = dict(mode='train', seed=settings['seed'], rank=int(best['rank']),
               learning_rate=float(best['learning_rate']), batch_size=int(best['microbatch']),
               workers=runtime['workers'], gradient_checkpointing=runtime['gradient_checkpointing'],
               train_path=str(train_path.resolve()), val_path=str((args.data_dir / 'val.jsonl').resolve()),
               baseline=settings['baseline'], max_steps=steps, eval_steps=100,
               epoch_steps=max(1, math.ceil(len(all_rows)/64)),
               save_steps=3000, output_dir=str(args.output_dir.resolve()), interactive=False)
    write_json(args.output_dir / 'selection.json', {'selected': best, 'source_questions': len(groups),
               'augmented_rows_per_epoch': len(all_rows), 'full_augmented_epochs': 2,
               'training_examples': len(complete_epochs), 'optimizer_steps': steps, 'job': job})
    print(f"Final training: {steps} optimizer steps (2 full augmented-data epochs, {len(complete_epochs)} rows), validation every augmented epoch", flush=True)
    result = execute_job(job, environment_for_gpu())
    if result.get('oom'):
        raise RuntimeError('Final training OOM; inspect logs')


if __name__ == '__main__':
    main()
