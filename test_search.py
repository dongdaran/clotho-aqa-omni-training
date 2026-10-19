"""CPU checks for budgets, augmented sampling, baseline compatibility and live logs."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from baseline_utils import load_baseline
from experiment_utils import (read_rows, source_groups, sample_stream, stratified_subset,
                              overfit_verdict, write_json, write_rows, file_digest, audit_data)
from monitoring import bridge_logs, scalar_record
from search_policy import MICROBATCHES, accumulation_for
from tune import run_attempt, execute_job, subset_baseline


def fixture_rows():
    rows = []
    for source in range(10):
        original = {'id': str(source), 'answer': 'yes' if source % 2 else 'no',
                    'question': f'Question {source}?', 'audio_path': 'fixture.wav'}
        rows.append(original)
        for kind in ('gain_-30dB', 'gain_-20dB', 'gain_-10dB', 'gain_+10dB', 'gain_+20dB', 'silence'):
            rows.append({**original, 'id': f'{source}_{kind}', 'source_id': str(source), 'augmentation': kind,
                         'answer': 'no' if kind == 'silence' else original['answer']})
    return rows


class DataAndBudgetTests(unittest.TestCase):
    def test_one_variant_per_source_visit_and_reproducibility(self):
        groups = source_groups(fixture_rows())
        stream = sample_stream(groups, 10000, seed=42)
        self.assertEqual(stream, sample_stream(groups, 10000, seed=42))
        for start in range(0, 10000, 10):
            chunk = stream[start:start+10]
            self.assertEqual(len({r.get('source_id', r['id']) for r in chunk}), 10)
        silence = sum(r.get('augmentation') == 'silence' for r in stream)
        self.assertTrue(0.12 < silence/len(stream) < 0.17)
        self.assertEqual(len({r.get('augmentation', 'original') for r in stream}), 7)

    def test_variants_need_a_source_and_correct_labels(self):
        rows = fixture_rows()
        rows[1]['answer'] = 'yes'  # source 0 is No
        with self.assertRaisesRegex(ValueError, 'Incorrect augmentation label'):
            source_groups(rows)
        with self.assertRaisesRegex(ValueError, 'exactly one original'):
            source_groups(fixture_rows()[1:])

    def test_stratified_validation_is_fixed_and_disjoint_from_training(self):
        rows = [r for r in fixture_rows() if not r.get('augmentation')]
        subset = stratified_subset(rows, 6, 42)
        self.assertEqual(sum(r['answer'] == 'yes' for r in subset), 3)
        self.assertEqual(subset, stratified_subset(rows[::-1], 6, 42))
        with self.assertRaises(ValueError):
            stratified_subset([r for r in rows if r['answer'] == 'yes'], 4, 42)

    def test_effective_batch(self):
        for size in MICROBATCHES:
            self.assertEqual(size*accumulation_for(size), 64)

    def test_overfit_gate(self):
        before, after = {'accuracy': 0.6}, {'accuracy': 0.9}
        self.assertTrue(overfit_verdict([.4, .4, .4, .2, .2, .2], before, after)['passed'])
        self.assertFalse(overfit_verdict([.4]*6, before, before)['passed'])
        self.assertTrue(overfit_verdict([.01]*6, {'accuracy': 1}, {'accuracy': 1})['passed'])

    def test_audit_rejects_split_leakage(self):
        import numpy as np
        import soundfile as sf
        from audio_augmentation import augment_records
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            paths = {}
            for split in ('train', 'val', 'test'):
                audio = base / f'{split}.wav'
                sf.write(audio, np.ones(160)*.01, 16000, subtype='FLOAT')
                paths[split] = [{'id': split, 'file_name': audio.name, 'audio_path': str(audio),
                                 'question': 'Are birds audible?', 'answer': 'yes', 'yes_votes': 1, 'no_votes': 0}]
            paths['train'] = augment_records(paths['train'], base / 'augmented')
            for split, rows in paths.items():
                write_rows(base / f'{split}.jsonl', rows)
            summary = audit_data(base)
            self.assertEqual(summary['train']['questions'], 7)
            paths['val'][0]['file_name'] = 'train.wav'
            write_rows(base / 'val.jsonl', paths['val'])
            with self.assertRaisesRegex(ValueError, 'leakage'):
                audit_data(base)


class BaselineTests(unittest.TestCase):
    def test_legacy_predictions_and_subset_use_the_same_questions(self):
        from qwen_train import metrics
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            rows = [{'id': 'a', 'answer': 'yes'}, {'id': 'b', 'answer': 'no'}]
            val = base / 'val.jsonl'
            write_rows(val, rows)
            predictions = [{**rows[0], 'prediction': 'yes'}, {**rows[1], 'prediction': None}]
            metric_path = base / 'val_predictions.metrics.json'
            write_json(metric_path, metrics(predictions))
            with self.assertRaisesRegex(ValueError, 'predictions TSV'):
                load_baseline(metric_path, val)
            tsv = base / 'val_predictions.tsv'
            with tsv.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'answer', 'prediction'], delimiter='\t')
                writer.writeheader()
                writer.writerows(predictions)
            self.assertEqual(load_baseline(metric_path, val), .5)
            self.assertEqual(subset_baseline(metric_path, rows), .5)
            write_json(metric_path, {**metrics(predictions), 'dataset_sha256': 'old'})
            with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
                load_baseline(metric_path, val)


class ProtocolTests(unittest.TestCase):
    def test_live_stdout_stderr_and_prune_handshake(self):
        code = ('import json,sys\n'
                'print("loss=0.2",flush=True)\n'
                'print("progress",file=sys.stderr,flush=True)\n'
                'print("AQA_EVENT "+json.dumps({"kind":"validation","step":100,"score":.7}),flush=True)\n'
                'decision=json.loads(input())\n'
                'assert decision["prune"] is True\n'
                'print("stopped",flush=True)\n')
        events = []
        def event(value):
            events.append(value)
            return {'prune': True}
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / 'console.log'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = run_attempt([sys.executable, '-u', '-c', code], dict(os.environ), log, event)
            self.assertEqual(result.returncode, 0)
            self.assertIn('loss=0.2', output.getvalue())
            self.assertIn('progress', output.getvalue())
            self.assertIn('stopped', output.getvalue())
            self.assertEqual(events[0]['step'], 100)
            self.assertIn('AQA_EVENT', log.read_text())

    def test_cached_jobs_and_oom(self):
        with tempfile.TemporaryDirectory() as root:
            job = {'output_dir': root, 'max_steps': 10, 'deadline': 123}
            with patch('tune.run_attempt', return_value=subprocess.CompletedProcess([], 75)):
                self.assertEqual(execute_job(job, {}), {'oom': True})
            write_json(Path(root) / 'completed.json', {'seconds_per_step': 2})
            self.assertEqual(execute_job({**job, 'deadline': 999}, {})['seconds_per_step'], 2)
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                execute_job({**job, 'max_steps': 20}, {})

    def test_legacy_scalar_parser(self):
        step, values = scalar_record({'global_step/max_steps': '60/2570', 'loss': .27, 'learning_rate': 2e-6})
        self.assertEqual(step, 60)
        self.assertEqual(values['loss'], .27)
        self.assertIsNone(scalar_record({'log_history': []}))

    def test_live_bridge_waits_for_complete_lines_and_keeps_source_unchanged(self):
        events = []
        class Writer:
            def __init__(self, path):
                pass
            def add_scalar(self, key, value, step):
                events.append((key, value, step))
            def flush(self):
                pass
            def close(self):
                pass
        stub = types.ModuleType('torch.utils.tensorboard')
        stub.SummaryWriter = Writer
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            source = base / 'trial_000/batch_16/logging.jsonl'
            source.parent.mkdir(parents=True)
            first = json.dumps({'global_step/max_steps': '10/100', 'loss': .3})+'\n'
            second = json.dumps({'global_step/max_steps': '20/100', 'loss': .2})
            source.write_text(first+second)
            sleeps = []
            def sleep(interval):
                sleeps.append(interval)
                if len(sleeps) == 1:
                    with source.open('a') as stream:
                        stream.write('\n')
                else:
                    raise KeyboardInterrupt
            with patch.dict(sys.modules, {'torch.utils.tensorboard': stub}), patch('monitoring.time.sleep', sleep):
                with self.assertRaises(KeyboardInterrupt):
                    bridge_logs(base)
            self.assertEqual(events, [('train/loss', .3, 10), ('train/loss', .2, 20)])
            self.assertEqual(source.read_text(), first+second+'\n')


@unittest.skipUnless(importlib.util.find_spec('optuna'), 'Optuna is not installed in this CPU environment')
class OptunaStorageTests(unittest.TestCase):
    def test_journal_restart_and_pruning(self):
        import optuna
        from tune import open_storage, trial_rows
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'study.journal'
            study = optuna.create_study(storage=open_storage(path), study_name='test', direction='maximize')
            trial = study.ask()
            trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
            trial.report(.8, 50)
            study.tell(trial, .8)
            reopened = optuna.load_study(storage=open_storage(path), study_name='test')
            self.assertEqual(len(reopened.trials), 1)
            self.assertEqual(trial_rows(reopened)[0]['value'], .8)


if __name__ == '__main__':
    unittest.main()
