"""Materialize train-only gain variants and one silent example per source question."""
from __future__ import annotations

import hashlib
from pathlib import Path

GAIN_DB = (-30, -20, -10, 10, 20)
MIN_DBFS = -65.0


def gain_waveform(waveform, gain_db: float):
    """Apply gain with an RMS floor, preserving values beyond digital full scale."""
    import numpy as np

    waveform = np.asarray(waveform, dtype=np.float64)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Audio must contain finite samples")
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    source_dbfs = float(20 * np.log10(rms)) if rms else None
    effective_gain = max(gain_db, MIN_DBFS - source_dbfs) if rms else gain_db
    amplified = waveform * 10 ** (effective_gain / 20)
    result = amplified.astype(np.float32)
    output_rms = float(np.sqrt(np.mean(result.astype(np.float64) ** 2)))
    return result, {
        "gain_db": gain_db, "effective_gain_db": effective_gain,
        "source_rms_dbfs": source_dbfs,
        "rms_dbfs": float(20 * np.log10(output_rms)) if output_rms else None,
        "samples_above_full_scale": int(np.count_nonzero(np.abs(result) > 1)),
    }


def augment_records(records: list[dict], audio_dir: Path) -> list[dict]:
    """Share rendered WAVs across questions about the same original audio."""
    import numpy as np
    import soundfile as sf

    audio_dir.mkdir(parents=True, exist_ok=True)
    by_audio = {}
    for record in records:
        by_audio.setdefault(record["audio_path"], []).append(record)
    augmented = []
    for source, questions in by_audio.items():
        waveform, sample_rate = sf.read(source, dtype="float64", always_2d=True)
        # Training uses mono audio; measure the floor on that same signal.
        waveform = waveform.mean(axis=1)
        key = hashlib.sha256(source.encode()).hexdigest()[:20]
        variants = []
        for gain in GAIN_DB:
            samples, metadata = gain_waveform(waveform, gain)
            path = audio_dir / f"{key}_gain_{gain:+d}dB.wav"
            sf.write(path, samples, sample_rate, subtype="FLOAT")
            variants.append((f"gain_{gain:+d}dB", path, metadata))
        silent_path = audio_dir / f"{key}_silence.wav"
        sf.write(silent_path, np.zeros_like(waveform), sample_rate, subtype="FLOAT")
        variants.append(("silence", silent_path, {"rms_dbfs": None}))
        for row in questions:
            augmented.append(row)
            for kind, path, metadata in variants:
                variant = {
                    **row, "id": f"{row['id']}_{kind}", "source_id": row["id"],
                    "audio_path": str(path.resolve()), "file_name": path.name,
                    "augmentation": kind, **metadata,
                }
                if kind == "silence":
                    variant["answer"] = "no"
                    # Synthetic labels have no human annotation votes.
                    variant["yes_votes"] = variant["no_votes"] = 0
                augmented.append(variant)
    return augmented
