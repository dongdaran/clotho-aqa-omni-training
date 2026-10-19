"""Prepare the official Clotho-AQA CSVs for binary audio question answering."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import difflib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
OFFICIAL_MD5 = {
    "train": "508513b91001ab17e641eaaa162ca045",
    "val": "41b8f6d906a57d7a5a4ee02bbcdefcc9",
    "test": "6fd0e75dffba561805fa37c668ce99a4",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_key(name: str) -> str:
    """Match Unicode/accent/punctuation variants without guessing sound content."""
    stem = Path(name).stem
    deaccented = unicodedata.normalize("NFKD", stem).casefold()
    return "".join(char for char in deaccented if char.isalnum() and not unicodedata.combining(char))


def find_audio(name: str, audio_dir: Path, aliases: dict[str, list[Path]]) -> Path:
    exact = audio_dir / name
    if exact.is_file():
        return exact
    candidates = aliases.get(audio_key(name), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous audio filename {name!r}: {[p.name for p in candidates]}")
    near = difflib.get_close_matches(name, [p.name for paths in aliases.values() for p in paths], n=5)
    raise FileNotFoundError(f"Missing Clotho-AQA audio: {exact}; closest extracted names: {near}")


def prepare(csv_dir: Path, audio_dir: Path, output_dir: Path, verify_md5: bool = True,
            augment_train: bool = True) -> dict:
    """Validate original splits and write one majority label per audio/question."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    seen_audio: dict[str, str] = {}
    aliases: dict[str, list[Path]] = defaultdict(list)
    for path in audio_dir.glob("*.wav"):
        aliases[audio_key(path.name)].append(path)
    for split in SPLITS:
        source = csv_dir / f"clotho_aqa_{split}.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        if verify_md5 and md5(source) != OFFICIAL_MD5[split]:
            raise ValueError(f"Official MD5 mismatch: {source}")
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        with source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"file_name", "QuestionText", "answer"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required CSV columns: {source}")
            for row in reader:
                label = row["answer"].strip().lower()
                if label not in {"yes", "no"}:
                    continue
                name = row["file_name"].strip()
                question = row["QuestionText"].strip()
                if not name or not question or Path(name).name != name:
                    raise ValueError(f"Invalid filename/question in {source}: {row!r}")
                groups[(name, question)].append(label)
        records = []
        for (name, question), votes in sorted(groups.items()):
            counts = Counter(votes)
            if counts["yes"] == counts["no"]:
                raise ValueError(f"Tied annotations: {split}, {name}, {question}")
            audio = find_audio(name, audio_dir, aliases)
            previous = seen_audio.setdefault(audio.name, split)
            if previous != split:
                raise ValueError(f"Audio appears in both {previous} and {split}: {audio.name}")
            records.append({
                "id": hashlib.sha256(f"{split}\0{name}\0{question}".encode()).hexdigest()[:20],
                "audio_path": str(audio.resolve()), "file_name": name,
                "question": question,
                "answer": "yes" if counts["yes"] > counts["no"] else "no",
                "yes_votes": counts["yes"], "no_votes": counts["no"],
            })
        if not records:
            raise ValueError(f"No binary questions in {source}")
        original_questions = len(records)
        if split == "train" and augment_train:
            from audio_augmentation import augment_records
            records = augment_records(records, output_dir / "augmented_audio" / split)
        target = output_dir / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        summaries[split] = {
            "original_questions": original_questions,
            "augmented_questions": len(records) - original_questions,
            "questions": len(records), "audio_files": len({r["file_name"] for r in records}),
            "answers": dict(Counter(r["answer"] for r in records)),
            "non_unanimous": sum(r["yes_votes"] and r["no_votes"] for r in records),
        }
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-md5", action="store_true", help="Only for nonofficial CSV variants")
    parser.add_argument("--no-augmentation", action="store_true", help="Keep the original train split")
    args = parser.parse_args()
    print(json.dumps(prepare(args.csv_dir, args.audio_dir, args.output_dir,
                             not args.skip_md5, not args.no_augmentation), indent=2))


if __name__ == "__main__":
    main()
