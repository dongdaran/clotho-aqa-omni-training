"""CPU-only checks for data grouping and binary evaluation rules."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from clotho_data import find_audio, audio_key, prepare
from qwen_train import metrics, normalize_answer


class DataPreparationTests(unittest.TestCase):
    def test_augmented_training_audio_and_labels(self):
        import numpy as np
        import soundfile as sf
        from audio_augmentation import GAIN_DB, gain_waveform

        # Quiet audio stops at the floor; positive gains preserve values above 1.
        quiet, info = gain_waveform(np.full(160, 0.01), -30)
        self.assertAlmostEqual(info["rms_dbfs"], -65, places=5)
        loud, info = gain_waveform(np.full(160, 0.5), 20)
        self.assertTrue(np.all(loud == 5))
        self.assertEqual(info["samples_above_full_scale"], 160)
        for gain in GAIN_DB:
            _, info = gain_waveform(np.full(160, 0.01), gain)
            self.assertAlmostEqual(info["rms_dbfs"], max(-65, -40 + gain), places=5)
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            source, audio, output = (base / name for name in ("source", "audio", "out"))
            source.mkdir()
            audio.mkdir()
            # FLOAT WAV must retain the amplified waveform, including peaks > 1.
            probe = audio / "gain_probe.wav"
            original = np.array([0.2, -0.5, 0.8, -0.1], dtype=np.float64)
            boosted, _ = gain_waveform(original, 20)
            sf.write(probe, boosted, 16000, subtype="FLOAT")
            restored, _ = sf.read(probe)
            np.testing.assert_allclose(restored, original * 10, rtol=1e-6)
            self.assertEqual(sf.info(probe).subtype, "FLOAT")
            for split in ("train", "val", "test"):
                sf.write(audio / f"{split}.wav", quiet, 16000, subtype="FLOAT")
                with (source / f"clotho_aqa_{split}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["file_name", "QuestionText", "answer"])
                    writer.writeheader()
                    for question in ("Are there birds?", "Is it outdoors?"):
                        writer.writerow({"file_name": f"{split}.wav", "QuestionText": question,
                                         "answer": "yes"})
            summary = prepare(source, audio, output, verify_md5=False)
            self.assertEqual([summary[s]["questions"] for s in ("train", "val", "test")], [14, 2, 2])
            rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len({r["id"] for r in rows}), 14)
            self.assertEqual(len(list((output / "augmented_audio/train").glob("*.wav"))), 6)
            for row in rows:
                samples, sr = sf.read(row["audio_path"])
                self.assertEqual((len(samples), sr), (160, 16000))
                if row.get("augmentation") == "silence":
                    self.assertEqual(row["answer"], "no")
                    self.assertTrue(np.all(samples == 0))
                else:
                    self.assertEqual(row["answer"], "yes")
                    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(samples ** 2)))
                    self.assertGreaterEqual(rms_dbfs, -65.00001)

    def test_unicode_audio_alias_is_unique(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root)
            variant = audio / "souffle_métallique.wav"
            variant.write_bytes(b"fixture")
            aliases = {audio_key(variant.name): [variant]}
            self.assertEqual(find_audio("souffle_me_tallique.wav", audio, aliases), variant)
            other = audio / "souffle_metallique.wav"
            other.write_bytes(b"fixture")
            aliases[audio_key(other.name)].append(other)
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                find_audio("souffle_me_tallique.wav", audio, aliases)

    def test_majority_and_audio_split_check(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            source, audio, output = (base / name for name in ("source", "audio", "out"))
            source.mkdir()
            audio.mkdir()
            for name in ("train.wav", "val.wav", "test.wav"):
                (audio / name).write_bytes(b"fixture")
            for split in ("train", "val", "test"):
                with (source / f"clotho_aqa_{split}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["file_name", "QuestionText", "answer"])
                    writer.writeheader()
                    for label in ("yes", "yes", "no"):
                        writer.writerow({"file_name": f"{split}.wav", "QuestionText": "Are there birds?",
                                         "answer": label})
                    writer.writerow({"file_name": f"{split}.wav", "QuestionText": "What sound?",
                                     "answer": "bird"})
            summary = prepare(source, audio, output, verify_md5=False, augment_train=False)
            self.assertEqual(summary["train"]["questions"], 1)
            self.assertEqual(summary["train"]["non_unanimous"], 1)
            self.assertIn('"answer": "yes"', (output / "train.jsonl").read_text())
            with (source / "clotho_aqa_val.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["file_name", "QuestionText", "answer"])
                writer.writeheader()
                writer.writerow({"file_name": "train.wav", "QuestionText": "Is it loud?", "answer": "no"})
            with self.assertRaisesRegex(ValueError, "Audio appears in both"):
                prepare(source, audio, output, verify_md5=False, augment_train=False)


class MetricTests(unittest.TestCase):
    def test_invalid_output_counts_as_wrong(self):
        rows = [{"answer": "yes", "prediction": "yes"},
                {"answer": "no", "prediction": None}]
        score = metrics(rows)
        self.assertEqual(score["accuracy"], 0.5)
        self.assertEqual(score["macro_f1"], 0.5)
        self.assertEqual(score["invalid_outputs"], 1)
        self.assertEqual(normalize_answer("Yes."), "yes")
        self.assertIsNone(normalize_answer("Maybe"))


if __name__ == "__main__":
    unittest.main()
