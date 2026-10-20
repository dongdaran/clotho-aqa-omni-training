"""Audit, profile, sanity-check and run a budgeted Optuna search on one GPU."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from baseline_utils import load_baseline
from experiment_utils import (audit_data, file_digest, read_rows, sample_stream, source_groups,
                              stratified_subset, write_json, write_rows)
from search_policy import MICROBATCHES
from swift_train import EVENT_PREFIX

STUDY_NAME = 'clotho-aqa-budgeted'
FIELDS = ['number', 'state', 'stage', 'learning_rate', 'rank', 'alpha', 'dropout', 'weight_decay',
          'value', 'full_val_value', 'best_step', 'stop_reason', 'microbatch', 'effective_batch_size',
          'adapter', 'intermediate_values']


def open_storage(path: Path):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))


def run_attempt(command: list[str], environment: dict, log_path: Path, on_event=None):
    """Stream both output channels, with synchronous pruning decisions on stdin."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1, errors='replace',
                              start_new_session=True) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(EVENT_PREFIX):
                        event = json.loads(line[len(EVENT_PREFIX):])
                        if event['kind'] == 'progress':
                            print(f"step {event['step']}/{event['max_steps']} | loss={event.get('loss', '-')} | "
                                  f"{event['seconds_per_step']:.2f}s/step | "
                                  f"ETA(train)={event['remaining_train_seconds']/60:.1f} min | "
                                  f"GPU={event.get('gpu_utilization_percent', '?')}%", flush=True)
                        decision = on_event(event) if on_event else None
                        if event['kind'] == 'validation':
                            process.stdin.write(json.dumps(decision or {'prune': False}) + '\n')
                            process.stdin.flush()
                    else:
                        print(line, end='', flush=True)
                return subprocess.CompletedProcess(command, process.wait())
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
                raise


def save_rows(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def trial_rows(study) -> list[dict]:
    rows = []
    for trial in study.trials:
        attrs = trial.user_attrs
        result = attrs.get('result', {})
        rows.append({
            'number': trial.number, 'state': trial.state.name, 'stage': attrs.get('stage', 'lr'),
            'learning_rate': trial.params.get('learning_rate'), 'rank': attrs.get('rank', 16),
            'alpha': 16, 'dropout': 0.05, 'weight_decay': 0.1, 'value': trial.value,
            'full_val_value': attrs.get('full_val_value', ''), 'best_step': result.get('best_step', ''),
            'stop_reason': result.get('stop_reason', ''), 'microbatch': result.get('batch_size', ''),
            'effective_batch_size': 64, 'adapter': result.get('adapter', ''),
            'intermediate_values': json.dumps(trial.intermediate_values, sort_keys=True),
        })
    return rows


def environment_for_gpu() -> dict:
    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    env.update(ENABLE_AUDIO_OUTPUT='0', USE_HF='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    return env


def execute_job(job: dict, env: dict, on_event=None) -> dict:
    output = Path(job['output_dir'])
    completed_path = output / 'completed.json'
    job_path = output / 'job.json'
    previous = json.loads(job_path.read_text()) if job_path.exists() else job
    if {k: v for k, v in previous.items() if k != 'deadline'} != {k: v for k, v in job.items() if k != 'deadline'}:
        raise ValueError(f'Job settings changed: {output}; use a new output directory')
    if completed_path.exists():
        return json.loads(completed_path.read_text())
    write_json(job_path, job)
    # Never append a new attempt to stale scalar history at the same steps.
    from recovery import latest_checkpoint
    if (output / 'step_metrics.jsonl').exists() and latest_checkpoint(output) is None:
        (output / 'step_metrics.jsonl').unlink()
    code = [sys.executable, '-u', str(Path(__file__).with_name('swift_train.py')),
            '--job', str(job_path.resolve())]
    process = run_attempt(code, env, output / 'console.log', on_event)
    if process.returncode == 75:
        return {'oom': True}
    process.check_returncode()
    result = json.loads((output / 'result.json').read_text())
    write_json(completed_path, result)
    return result


def subset_baseline(metrics_path: Path, subset: list[dict]) -> float:
    from qwen_train import metrics
    path = metrics_path.with_name(metrics_path.name.removesuffix('.metrics.json') + '.tsv')
    if not path.is_file():
        raise ValueError('Baseline predictions TSV is needed to score the same validation subset; rerun baseline')
    with path.open(newline='') as stream:
        predictions = list(csv.DictReader(stream, delimiter='\t'))
    for row in predictions:
        row['prediction'] = row['prediction'] or None
    ids = {r['id']: r['answer'] for r in subset}
    selected = [r for r in predictions if r['id'] in ids]
    if len(selected) != len(ids) or {r['id']: r['answer'] for r in selected} != ids:
        raise ValueError('Baseline predictions do not cover the validation subset')
    value = metrics(selected)['macro_f1']
    if value is None:
        raise ValueError('Validation subset must include both labels')
    return value


def preparation(data_dir, output, settings, baseline_path, env, phase=None):
    phase = phase or (lambda number, name: None)
    phase(1, 'data_audit')
    print('[1/7] Auditing labels, audio lengths and train/val/test separation', flush=True)
    audit = audit_data(data_dir)
    write_json(output / 'data_audit.json', audit)
    training, val = read_rows(data_dir / 'train.jsonl'), read_rows(data_dir / 'val.jsonl')
    groups = source_groups(training)
    subset = stratified_subset(val, settings['val_size'], settings['seed'])
    subset_path = output / 'val_subset.jsonl'
    write_rows(subset_path, subset)
    baseline = subset_baseline(baseline_path, subset)
    schedule = sample_stream(groups, settings['max_steps'] * 64, settings['seed'])
    train_path = output / 'train_sampled.jsonl'
    write_rows(train_path, schedule)
    common = {
        'seed': settings['seed'], 'rank': 16, 'learning_rate': 5e-5,
        'train_path': str(train_path.resolve()), 'val_path': str(subset_path.resolve()),
        'baseline': baseline, 'eval_steps': settings['eval_steps'], 'min_steps': settings['min_steps'],
        'patience': 2, 'interactive': False,
    }
    print('[2/7] Measuring OOM limits and data-loader/checkpointing throughput', flush=True)
    phase(2, 'performance_profiles')
    benchmarks = []
    # Use long originals to avoid selecting a batch that only fits short clips.
    import soundfile as sf
    longest = sorted([next(r for r in group if not r.get('augmentation')) for group in groups],
                     key=lambda r: (sf.info(r['audio_path']).duration, len(r['question'])), reverse=True)[:64]
    profile_path = output / 'profile_train.jsonl'
    write_rows(profile_path, (longest * ((settings['profile_steps'] * 64 + len(longest)-1)//len(longest)))
               [:settings['profile_steps'] * 64])
    profile_common = {**common, 'mode': 'profile', 'max_steps': settings['profile_steps'],
                      'train_path': str(profile_path.resolve()), 'val_path': None}
    for batch in (16,):
        job = {**profile_common, 'batch_size': batch, 'workers': 0, 'gradient_checkpointing': True,
               'output_dir': str((output / 'preflight' / f'batch_{batch}').resolve())}
        result = execute_job(job, env)
        if result.get('oom'):
            print(f'Preflight OOM: batch={batch}, accumulation=4; fixed batch requested, stopping', flush=True)
            continue
        benchmarks.append(result)
        chosen_batch = batch
        break
    else:
        raise RuntimeError('OOM at fixed microbatch 16 / gradient accumulation 4')
    profile_failures = []
    for profile_index in range(2):
        workers = 2 if profile_index == 0 else min(benchmarks, key=lambda item: item['seconds_per_step'])['workers']
        checkpointing = profile_index == 0
        job = {**profile_common, 'batch_size': chosen_batch, 'workers': workers,
               'gradient_checkpointing': checkpointing,
               'output_dir': str((output / 'preflight' / f'workers_{workers}_checkpoint_{checkpointing}').resolve())}
        try:
            result = execute_job(job, env)
        except subprocess.CalledProcessError as error:
            profile_failures.append({'workers': workers, 'gradient_checkpointing': checkpointing,
                                     'exit_code': error.returncode, 'log': job['output_dir'] + '/console.log'})
            print('Optional performance profile failed; keeping the working configuration. See its console.log.', flush=True)
            continue
        if not result.get('oom'):
            benchmarks.append(result)
    fastest = min(benchmarks, key=lambda item: item['seconds_per_step'])
    write_json(output / 'performance.json', {'profiles': benchmarks, 'failed_profiles': profile_failures, 'selected': fastest,
               'estimated_train_seconds_per_trial': fastest['seconds_per_step'] * settings['max_steps']})
    runtime = {key: fastest[key] for key in ('batch_size', 'workers', 'gradient_checkpointing')}
    print(f"Selected batch={runtime['batch_size']} workers={runtime['workers']} "
          f"checkpoint={runtime['gradient_checkpointing']}; {fastest['seconds_per_step']:.2f}s/step; "
          f"{settings['max_steps'] * fastest['seconds_per_step']/60:.1f} min/trial plus validation", flush=True)
    print('[3/7] Checking answer masks, frozen encoders and learning on 32 source questions', flush=True)
    phase(3, 'tiny_overfit_and_audio_ablation')
    originals = [next(r for r in group if not r.get('augmentation')) for group in groups]
    tiny = stratified_subset(originals, min(32, len(originals)), settings['seed'])
    tiny_val = output / 'tiny_val.jsonl'
    tiny_train = output / 'tiny_train.jsonl'
    write_rows(tiny_val, tiny)
    write_rows(tiny_train, (tiny * ((64+len(tiny)-1)//len(tiny)))[:64])
    tiny_ids = {r['id'] for r in tiny}
    silence = []
    for group in groups:
        original = next(r for r in group if not r.get('augmentation'))
        if original['id'] in tiny_ids:
            muted = next(r for r in group if r.get('augmentation') == 'silence')
            silence.append({**muted, 'id': original['id']})
    silence_path = output / 'tiny_silence.jsonl'
    write_rows(silence_path, silence)
    overfit = execute_job({**common, **runtime, 'mode': 'overfit', 'max_steps': settings['overfit_steps'],
                          'train_path': str(tiny_train.resolve()), 'val_path': str(tiny_val.resolve()),
                          'ablation_path': str(silence_path.resolve()),
                          'learning_rate': 1e-4,
                          'output_dir': str((output / 'overfit').resolve())}, env)
    if overfit.get('oom'):
        raise RuntimeError('Overfit check ran out of memory; review performance.json before searching')
    if not overfit['passed']:
        raise RuntimeError('Tiny-data learning check did not pass. Inspect overfit/result.json and label_check.json before searching.')
    phase(4, 'augmentation_sampling')
    print(f"[4/7] {len(groups)} source questions; one random variant per source visit; "
          f"{settings['max_steps']} updates × 64 samples", flush=True)
    return common, runtime


def run(data_dir: Path, output: Path, baseline_path: Path, settings: dict):
    import optuna
    baseline_full = load_baseline(baseline_path, data_dir / 'val.jsonl')
    manifest = {**settings, 'framework': 'ms-swift==3.12.0', 'pipeline_version': 3,
                'selection_metric': 'macro_f1', 'optimizer': 'adamw_torch',
                'alpha': 16, 'dropout': 0.05, 'weight_decay': 0.1, 'effective_batch_size': 64,
                'baseline': baseline_full,
                'data': {s: file_digest(data_dir / f'{s}.jsonl') for s in ('train', 'val', 'test')}}
    config = output / 'search_config.json'
    if config.exists() and json.loads(config.read_text()) != manifest:
        raise ValueError('Data/settings changed; use a new output directory')
    if not config.exists() and (output / 'trials.tsv').exists():
        raise ValueError('Legacy results found; use a new output directory')
    write_json(config, manifest)
    env = environment_for_gpu()
    started = time.time()
    storage = open_storage(output / 'study.journal')
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction='maximize', load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=settings['seed'], n_startup_trials=1),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=1,
                                   n_warmup_steps=settings['min_steps'], interval_steps=settings['eval_steps']))
    def phase(number, name):
        study.set_user_attr('pipeline_phase', {'number': number, 'name': name})
        write_json(output / 'current.json', {'state': 'PREPARING', 'phase': number, 'name': name})
    common, runtime = preparation(data_dir, output, settings, baseline_path, env, phase)
    deadline = started + settings['timeout_seconds'] if settings['timeout_seconds'] else None
    # A crashed child is restarted as a fresh trial with the same parameters.
    for old in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(old.number, state=optuna.trial.TrialState.FAIL)
        study.enqueue_trial(old.params, user_attrs={'stage': old.user_attrs.get('stage', 'lr'),
                                                   'rank': old.user_attrs.get('rank', 16),
                                                   'resume_trial_number': old.user_attrs.get('resume_trial_number', old.number)})
    if not study.trials:
        for lr in (5e-5, 1e-4):
            study.enqueue_trial({'learning_rate': lr}, user_attrs={'stage': 'lr', 'rank': 16})
    print('[5/7] Fixed small validation set every 50 steps; full validation for finalists', flush=True)
    phase(5, 'fixed_validation_subset')
    print('[6/7] Optuna: learning rate first at rank 16; rank 32 only after a promising result', flush=True)
    phase(6, 'optuna_search')
    print('[7/7] Live stdout/stderr, TensorBoard and Optuna progress reporting active', flush=True)

    def objective(trial):
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
        rank = trial.user_attrs.get('rank', 16)
        trial.set_user_attr('rank', rank)
        trial.set_user_attr('stage', trial.user_attrs.get('stage', 'lr'))
        trial.set_user_attr('baseline_subset', common['baseline'])

        def events(event):
            write_json(output / 'current.json', {'trial': trial.number, 'rank': rank, 'learning_rate': lr, **event})
            if event['kind'] == 'progress':
                trial.set_user_attr('progress', {k: event[k] for k in ('step', 'max_steps', 'remaining_train_seconds',
                                                                     'seconds_per_step', 'loss') if k in event})
            if event['kind'] == 'validation':
                trial.report(event['score'], step=event['step'])
                if event['step'] >= settings['min_steps'] and event['step'] < settings['max_steps']:
                    if trial.should_prune():
                        return {'prune': True, 'reason': 'optuna_pruned'}
                return {'prune': False}

        for batch in (16,):
            job = {**common, **runtime, 'batch_size': batch, 'mode': 'search', 'interactive': True,
                   'rank': rank, 'learning_rate': lr, 'max_steps': settings['max_steps'],
                   'deadline': deadline,
                   'output_dir': str((output / f"trial_{trial.user_attrs.get('resume_trial_number', trial.number):03d}" / f'batch_{batch}').resolve())}
            print(f"Trial {trial.number}: rank={rank} LR={lr:.3g} batch={batch} accumulation={64//batch}", flush=True)
            result = execute_job(job, env, events)
            if result.get('oom'):
                continue
            trial.set_user_attr('result', result)
            if result['stop_reason'] == 'optuna_pruned':
                raise optuna.TrialPruned()
            if not result.get('best'):
                raise RuntimeError('No validation checkpoint was produced')
            return result['best']['macro_f1']
        raise RuntimeError('OOM even at microbatch 1')

    def completed(study, trial):
        save_rows(output / 'trials.tsv', trial_rows(study))
        print(f'Trial {trial.number} {trial.state.name}: {trial.value}', flush=True)

    def remaining_time():
        return None if deadline is None else max(0, deadline-time.time())

    lr_done = [t for t in study.trials if t.user_attrs.get('stage', 'lr') == 'lr'
               and t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    if remaining_time() is None or remaining_time() > 0:
        study.optimize(objective, n_trials=max(0, settings['trials']-len(lr_done)),
                       timeout=remaining_time(), callbacks=[completed])
    candidates = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rank_done = any(t.user_attrs.get('stage') == 'rank' and t.state in
                    (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
    if candidates and not rank_done and settings['try_rank32'] and (remaining_time() is None or remaining_time() > 0):
        best_lr = max(candidates, key=lambda t: t.value)
        if best_lr.value > common['baseline']:
            rank_waiting = any(t.state == optuna.trial.TrialState.WAITING and t.user_attrs.get('stage') == 'rank'
                               for t in study.trials)
            if not rank_waiting:
                study.enqueue_trial({'learning_rate': best_lr.params['learning_rate']}, user_attrs={'stage': 'rank', 'rank': 32})
            study.optimize(objective, n_trials=1, timeout=remaining_time(), callbacks=[completed])
    candidates = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                        key=lambda t: t.value, reverse=True)[:2]
    if not candidates:
        raise RuntimeError('No completed trial; inspect console logs and Optuna Dashboard')
    for candidate in candidates:
        study.set_user_attr('pipeline_phase', {'number': 7, 'name': 'full_validation_reranking'})
        evaluation = output / f'trial_{candidate.number:03d}' / 'full_val'
        metrics_path = evaluation / 'evaluation.json'
        if not metrics_path.exists():
            command = [sys.executable, '-u', str(Path(__file__).with_name('swift_train.py')),
                       '--adapter', candidate.user_attrs['result']['adapter'], '--data-dir', str(data_dir.resolve()),
                       '--output-dir', str(evaluation.resolve()), '--split', 'val']
            run_attempt(command, env, evaluation / 'console.log').check_returncode()
        score = json.loads(metrics_path.read_text())['macro_f1']
        # Public live Trial handle lets the dashboard show full-validation reranking.
        optuna.trial.Trial(study, candidate._trial_id).set_user_attr('full_val_value', score)
    rows = trial_rows(study)
    eligible = [r for r in rows if r['state'] == 'COMPLETE' and r['full_val_value'] != '']
    best = max(eligible, key=lambda r: r['full_val_value'])
    save_rows(output / 'trials.tsv', rows)
    save_rows(output / 'best.tsv', [best])
    write_json(output / 'current.json', {'state': 'FINISHED', 'best': best, 'baseline_full': baseline_full})
    study.set_user_attr('pipeline_phase', {'number': 7, 'name': 'finished'})
    print(f"Selected trial {best['number']}: full val={best['full_val_value']:.4f}, baseline={baseline_full:.4f}", flush=True)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/optuna_budgeted'))
    parser.add_argument('--baseline-metrics', type=Path, default=Path('results/baseline_3b/val_predictions.metrics.json'))
    parser.add_argument('--trials', type=int, default=3, help='Learning-rate trials at rank 16')
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--min-steps', type=int, default=100)
    parser.add_argument('--eval-steps', type=int, default=50)
    parser.add_argument('--val-size', type=int, default=200)
    parser.add_argument('--profile-steps', type=int, default=10)
    parser.add_argument('--overfit-steps', type=int, default=30)
    parser.add_argument('--timeout-seconds', type=int, default=21600)
    parser.add_argument('--skip-rank32', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not (args.trials >= 1 and args.max_steps >= args.min_steps >= args.eval_steps >= 1
            and args.profile_steps >= 4 and args.overfit_steps >= 4 and args.val_size >= 2
            and args.timeout_seconds >= 0):
        parser.error('Invalid trial/step/time budget')
    settings = {key: value for key, value in vars(args).items()
                if key not in ('data_dir', 'output_dir', 'baseline_metrics', 'skip_rank32')}
    settings['try_rank32'] = not args.skip_rank32
    try:
        run(args.data_dir, args.output_dir, args.baseline_metrics, settings)
    except KeyboardInterrupt:
        write_json(args.output_dir / 'current.json', {'state': 'INTERRUPTED'})
        raise
    except Exception as error:
        write_json(args.output_dir / 'current.json', {'state': 'FAILED', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
