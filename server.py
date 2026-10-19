"""Run data preparation, baseline, Optuna search and test on an A100 80GB."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, '-u', str(ROOT / script), *map(str, args)], check=True)


def prepare(data):
    source, audio = data / 'source', data / 'audio'
    source.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)
    for name in ['audio_files.zip', *(f'clotho_aqa_{s}.csv' for s in ('train', 'val', 'test'))]:
        target = source / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.part')
            subprocess.run(['curl', '-fL', '--retry', '5',
                            f'https://zenodo.org/records/6473207/files/{name}?download=1',
                            '-o', str(temporary)], check=True)
            temporary.replace(target)
    with (source / 'audio_files.zip').open('rb') as stream:
        if hashlib.file_digest(stream, 'md5').hexdigest() != '48c2c46ccd71dc06121228e2ed68d6b9':
            raise ValueError('Official audio ZIP checksum mismatch')
    with zipfile.ZipFile(source / 'audio_files.zip') as archive:
        wavs = [n for n in archive.namelist() if n.lower().endswith('.wav')]
        if not wavs or len({Path(n).name for n in wavs}) != len(wavs):
            raise ValueError('Empty audio ZIP or duplicate WAV names')
        for name in wavs:
            target = audio / Path(name).name
            if target.exists() and target.stat().st_size == archive.getinfo(name).file_size:
                continue
            temporary = target.with_suffix('.part')
            with archive.open(name) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            temporary.replace(target)
    run('clotho_data.py', '--csv-dir', source, '--audio-dir', audio,
        '--output-dir', data / 'processed')


def check_gpu():
    # Exit the probe process before training so it holds no GPU context/VRAM.
    subprocess.run([sys.executable, '-c',
                    'from gpu_runtime import configure_cuda; import torch; '
                    'configure_cuda(); gpu = torch.cuda.get_device_properties(0); '
                    'print(f"GPU: {gpu.name}; VRAM: {gpu.total_memory / 2**30:.1f} GiB", flush=True); '
                    'assert gpu.total_memory >= 70 * 2**30, "This launcher targets a single 80GB GPU"'],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'prepare', 'baseline', 'search', 'test'), default='all')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/a100_macro_f1')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--eval-steps', type=int, default=50)
    parser.add_argument('--min-steps', type=int, default=100)
    parser.add_argument('--timeout-seconds', type=int, default=21600)
    parser.add_argument('--skip-rank32', action='store_true')
    parser.add_argument('--no-monitor', action='store_true')
    args = parser.parse_args()
    if not (args.trials >= 1 and args.max_steps >= args.min_steps >= args.eval_steps >= 1
            and args.timeout_seconds >= 0):
        parser.error('Invalid trial/step/time budget')
    data, output = args.data_root.resolve(), args.output_root.resolve()
    processed, baseline = data / 'processed', output / 'baseline/val_predictions.metrics.json'
    search = output / 'search'
    if args.stage != 'prepare':
        check_gpu()
    if args.stage in ('all', 'prepare'):
        prepare(data)
    if args.stage in ('all', 'baseline'):
        if baseline.exists():
            from baseline_utils import load_baseline
            load_baseline(baseline, processed / 'val.jsonl')
        else:
            run('qwen_train.py', '--mode', 'baseline', '--data-dir', processed,
                '--output-dir', baseline.parent)
    if args.stage in ('all', 'search'):
        if not args.no_monitor:
            from monitoring import start_monitoring
            start_monitoring(search, show_colab=False)
        flags = ['--skip-rank32'] if args.skip_rank32 else []
        run('tune.py', '--data-dir', processed, '--baseline-metrics', baseline,
            '--output-dir', search, '--trials', args.trials, '--max-steps', args.max_steps,
            '--eval-steps', args.eval_steps, '--min-steps', args.min_steps,
            '--timeout-seconds', args.timeout_seconds, *flags)
    if args.stage in ('all', 'test'):
        run('finalize.py', '--data-dir', processed, '--best-tsv', search / 'best.tsv',
            '--output-dir', output / 'test')
    print(f'Finished {args.stage}. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
