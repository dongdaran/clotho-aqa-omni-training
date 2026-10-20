import tempfile
import unittest
from pathlib import Path
from recovery import prediction_identity, load_predictions, save_predictions, latest_checkpoint


class RecoveryTests(unittest.TestCase):
    def test_prediction_resume_checks_identity_and_prefix(self):
        rows = [{'id': 'a', 'answer': 'yes'}, {'id': 'b', 'answer': 'no'}]
        identity = prediction_identity(rows, 'base-model')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial.json'
            self.assertEqual(load_predictions(path, identity, rows), [])
            first = [{**rows[0], 'prediction': None, 'raw_output': 'unknown'}]
            save_predictions(path, identity, first)
            self.assertEqual(load_predictions(path, identity, rows), first)
            with self.assertRaises(ValueError):
                load_predictions(path, prediction_identity(rows, 'different-model'), rows)
            with self.assertRaises(ValueError):
                load_predictions(path, identity, rows[::-1])

    def test_incomplete_newer_checkpoint_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / 'checkpoint-50'
            good.mkdir()
            for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'aqa_state.json'):
                (good / name).touch()
            broken = root / 'checkpoint-100'
            broken.mkdir()
            (broken / 'trainer_state.json').touch()
            self.assertEqual(latest_checkpoint(root), good)
            (good / 'optimizer.pt').unlink()
            self.assertIsNone(latest_checkpoint(root))


if __name__ == '__main__':
    unittest.main()
