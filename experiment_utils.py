"""Deterministic data selection, validation and experiment artifacts."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path


def read_rows(path: Path, allow_repeated: bool = False) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or (not allow_repeated and len({r["id"] for r in rows}) != len(rows)):
        raise ValueError(f"Empty data or duplicate IDs: {path}")
    if any(r["answer"] not in ("yes", "no") for r in rows):
        raise ValueError(f"Nonbinary labels: {path}")
    return rows


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_groups(rows: list[dict]) -> list[list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("source_id", row["id"])].append(row)
    result = []
    for source_id, variants in sorted(groups.items()):
        originals = [r for r in variants if r["id"] == source_id and not r.get("augmentation")]
        if len(originals) != 1:
            raise ValueError(f"Expected exactly one original row for {source_id}")
        original = originals[0]
        kinds = [r.get("augmentation", "original") for r in variants]
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"Duplicate variant for {source_id}")
        for row in variants:
            if row["question"] != original["question"]:
                raise ValueError("Augmented questions must match their source")
            expected = "no" if row.get("augmentation") == "silence" else original["answer"]
            if row["answer"] != expected:
                raise ValueError("Incorrect augmentation label")
        result.append(sorted(variants, key=lambda r: r.get("augmentation", "")))
    return result


def sample_stream(groups: list[list[dict]], count: int, seed: int) -> list[dict]:
    """Visit every source once per shuffled pass, drawing one of its variants."""
    if not groups or count < 1:
        raise ValueError("A positive sample budget and nonempty source groups are required")
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        order = list(range(len(groups)))
        rng.shuffle(order)
        for index in order:
            result.append(rng.choice(groups[index]))
            if len(result) == count:
                break
    return result


def stratified_subset(rows: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r['id'])
    by_label = {label: [r for r in rows if r["answer"] == label] for label in ("yes", "no")}
    if not all(by_label.values()):
        raise ValueError("Validation must contain both labels")
    if size < 2:
        raise ValueError("Validation subset size must be >= 2")
    if size >= len(rows):
        return sorted(rows, key=lambda r: r["id"])
    selected = []
    for label, quota in (("yes", size // 2), ("no", size - size // 2)):
        selected.extend(rng.sample(by_label[label], min(quota, len(by_label[label]))))
    selected_ids = {r["id"] for r in selected}
    remaining = [r for r in rows if r["id"] not in selected_ids]
    selected.extend(rng.sample(remaining, size - len(selected)))
    return sorted(selected, key=lambda r: r["id"])


def audit_data(data_dir: Path) -> dict:
    import soundfile as sf
    from clotho_data import audio_key

    summary, seen = {}, {}
    for split in ("train", "val", "test"):
        rows = read_rows(data_dir / f"{split}.jsonl")
        originals = [r for r in rows if not r.get("augmentation")]
        if split == "train":
            groups = source_groups(rows)
            expected = {"original", "silence", "gain_-30dB", "gain_-20dB", "gain_-10dB",
                        "gain_+10dB", "gain_+20dB"}
            for group in groups:
                if {r.get("augmentation", "original") for r in group} != expected:
                    raise ValueError("Prepare the five gain variants and silence before running this experiment")
        elif len(originals) != len(rows):
            raise ValueError(f"{split} must remain unaugmented")
        for row in rows:
            if not Path(row["audio_path"]).is_file():
                raise FileNotFoundError(row["audio_path"])
        audio = {r["audio_path"]: r for r in originals}
        durations = []
        sample_rates = Counter()
        for path, row in audio.items():
            key = str(Path(path).resolve())
            content_key = "sha256:" + file_digest(Path(path))
            previous = seen.setdefault(content_key, split)
            if previous == split:
                previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError(f"Audio leakage between {previous} and {split}: {path}")
            info = sf.info(path)
            if info.frames <= 0:
                raise ValueError(f"Empty audio: {path}")
            durations.append(info.duration)
            sample_rates[info.samplerate] += 1
        summary[split] = {
            "questions": len(rows), "original_questions": len(originals),
            "answers": dict(Counter(r["answer"] for r in rows)), "audio_files": len(audio),
            "duration_seconds": {"min": min(durations), "max": max(durations),
                                 "mean": sum(durations) / len(durations)},
            "sample_rates": dict(sample_rates), "sha256": file_digest(data_dir / f"{split}.jsonl"),
        }
    return summary


def overfit_verdict(losses: list[float], before: dict, after: dict) -> dict:
    import statistics
    if len(losses) < 4:
        raise ValueError("Overfit check requires at least four logged losses")
    first = statistics.median(losses[:3])
    last = statistics.median(losses[-3:])
    # A nearly solved tiny task has little accuracy/loss headroom left.
    solved = after["accuracy"] >= 0.95 and last <= first * 1.10
    learned = last < first * 0.95 and after["accuracy"] >= before["accuracy"]
    return {"passed": solved or learned, "first_loss": first, "last_loss": last,
            "before": before, "after": after}
