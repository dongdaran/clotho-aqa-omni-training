"""Evaluate Qwen Omni checkpoints on StemFoley with three closed-set rules."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
REVISION = "dab3181ae9543d1c9ead49c36bfcae6e649e9e66"
PROMPT = "Does this audio contain the sound of {caption}? Respond using only one word: yes or no."
METHODS = ("tad_token_set", "yes_probability_over_50", "single_token")


def answer_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    groups = []
    for word in ("yes", "no"):
        ids: set[int] = set()
        for prefix in ("", " "):
            for variant in (word, word.title(), word.upper()):
                try:
                    encoded = tokenizer.encode(prefix + variant, add_special_tokens=False)
                except TypeError:
                    encoded = tokenizer.encode(prefix + variant, bos=False, eos=False)
                ids.update(encoded)
        if not ids:
            raise ValueError(f"No token IDs for {word!r} in tokenizer")
        groups.append(sorted(ids))
    if set(groups[0]) & set(groups[1]):
        raise ValueError("Yes/no token ID groups overlap")
    return groups[0], groups[1]


def load_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    with (dataset_dir / "test_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    rows = []
    for item in labels:
        audio_path = dataset_dir / item["audio_path"]
        target = item["target_audible"].strip().lower()
        if target not in {"yes", "no"}:
            raise ValueError(f"Invalid target_audible for {item['sample_id']}: {target!r}")
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        rows.append({
            "id": item["sample_id"], "answer": target,
            "question": PROMPT.format(caption=item["target"].strip()),
            "audio_path": item["audio_path"], "_resolved_audio_path": str(audio_path),
            "difficulty": item["difficulty"],
            "in_final_v3": item["in_final_v3"].strip().lower() == "true",
        })
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample_id in StemFoley labels")
    return rows


def batch_inputs(rows, processor, device, dtype):
    from qwen_train import evaluation_audio
    from tqdm.auto import tqdm
    import torch
    sample_rate = processor.feature_extractor.sampling_rate
    messages = [[
            {"role": "system", "content": [{"type": "text", "text": "Listen to the audio and answer the question with exactly Yes or No."}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": "audio.wav"},
                {"type": "text", "text": row["question"]},
            ]},
        ] for row in rows]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
             for message in messages]
    processor.tokenizer.padding_side = "left"
    extractor = processor.feature_extractor

    class DeviceFeatureExtractor:
        def __init__(self, wrapped, target_device):
            self.wrapped = wrapped
            self.target_device = target_device

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def __call__(self, *args, **kwargs):
            kwargs["device"] = self.target_device
            return self.wrapped(*args, **kwargs)

    processor.feature_extractor = DeviceFeatureExtractor(extractor, str(device))
    try:
        feature_batches, mask_batches, token_rows = [], [], []
        feature_progress = tqdm(total=len(rows), desc="Audio features", unit="item",
                                leave=False, dynamic_ncols=True)
        # The Qwen processor pads every audio clip to its 5-minute maximum.
        # Chunk feature extraction to cap temporary CPU allocations while keeping
        # the requested (up to 128) model-forward batch intact.
        feature_chunk_size = 16
        for start in range(0, len(rows), feature_chunk_size):
            end = min(len(rows), start + feature_chunk_size)
            waveforms = [evaluation_audio(row["_resolved_audio_path"], sample_rate) for row in rows[start:end]]
            encoded = processor(text=texts[start:end], audio=waveforms, sampling_rate=sample_rate,
                                padding=True, return_tensors="pt")
            feature_batches.append(encoded["input_features"])
            mask_batches.append(encoded["feature_attention_mask"])
            for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
                ids = ids[mask.bool()]
                token_rows.append({"input_ids": ids.tolist(), "attention_mask": [1] * len(ids)})
            feature_progress.update(end - start)
        feature_progress.close()
    finally:
        processor.feature_extractor = extractor
    text_batch = processor.tokenizer.pad(token_rows, padding=True, return_tensors="pt")
    encoded = {
        **text_batch,
        "input_features": torch.cat(feature_batches, dim=0),
        "feature_attention_mask": torch.cat(mask_batches, dim=0),
    }
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in encoded.items()}


def token_scores(logits, yes_ids: list[int], no_ids: list[int], single_yes: int,
                 single_no: int) -> list[dict[str, Any]]:
    import torch
    logits = logits.float()
    yes = torch.logsumexp(logits[:, yes_ids], dim=-1)
    no = torch.logsumexp(logits[:, no_ids], dim=-1)
    pair_probability = torch.softmax(torch.stack((yes, no), dim=-1), dim=-1)[:, 0]
    single_yes_score = logits[:, single_yes]
    single_no_score = logits[:, single_no]
    single_probability = torch.softmax(torch.stack((single_yes_score, single_no_score), dim=-1), dim=-1)[:, 0]
    output = []
    for index in range(logits.shape[0]):
        # Rule 1 is the supplied TAD logsumexp comparison. Rule 2 applies the
        # requested 50% threshold after normalizing those same two group scores.
        tad_pred = "yes" if yes[index].item() > no[index].item() else "no"
        probability = pair_probability[index].item()
        threshold_pred = "yes" if probability > 0.5 else "no"
        output.append({
            "tad_token_set_prediction": tad_pred,
            "tad_yes_logit": yes[index].item(), "tad_no_logit": no[index].item(),
            "tad_yes_no_probability": probability,
            "yes_probability_over_50_prediction": threshold_pred,
            "single_token_prediction": "yes" if single_yes_score[index].item() > single_no_score[index].item() else "no",
            "single_yes_token_logit": single_yes_score[index].item(),
            "single_no_token_logit": single_no_score[index].item(),
            "single_yes_probability": single_probability[index].item(),
        })
    return output


def metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    n = len(rows)
    correct = sum(row["answer"] == row[prediction_key] for row in rows)
    class_f1 = []
    for label in ("yes", "no"):
        tp = sum(row["answer"] == label and row[prediction_key] == label for row in rows)
        fp = sum(row["answer"] != label and row[prediction_key] == label for row in rows)
        fn = sum(row["answer"] == label and row[prediction_key] != label for row in rows)
        denom = 2 * tp + fp + fn
        class_f1.append(2 * tp / denom if denom else 0.0)
    return {"n": n, "accuracy": correct / n if n else None,
            "macro_f1": sum(class_f1) / 2 if n else None,
            "yes_f1": class_f1[0] if n else None, "no_f1": class_f1[1] if n else None}


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_resume(path: Path, run_key: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_key") != run_key:
        return {}
    expected = {row["id"] for row in rows}
    return {key: value for key, value in payload.get("predictions", {}).items() if key in expected}


def write_model_outputs(output_dir: Path, model_name: str, rows: list[dict[str, Any]],
                        predictions: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        scores = predictions[row["id"]]
        flat.append({**row, **scores})
    fields = ["id", "answer", "difficulty", "in_final_v3", "question", "audio_path",
              "tad_token_set_prediction", "tad_yes_logit", "tad_no_logit", "tad_yes_no_probability",
              "yes_probability_over_50_prediction", "single_token_prediction",
              "single_yes_token_logit", "single_no_token_logit", "single_yes_probability"]
    with (output_dir / f"{model_name}.predictions.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    records = []
    key_for_method = {
        "tad_token_set": "tad_token_set_prediction",
        "yes_probability_over_50": "yes_probability_over_50_prediction",
        "single_token": "single_token_prediction",
    }
    for subset, subset_rows in (("all", rows), ("in_final_v3", [r for r in rows if r["in_final_v3"]])):
        for method in METHODS:
            records.append({"model": model_name, "subset": subset, "method": method,
                            **metrics([{**r, **predictions[r["id"]]} for r in subset_rows], key_for_method[method])})
    atomic_json(output_dir / f"{model_name}.metrics.json", {"metadata": metadata, "results": records})
    return records


def evaluate_all(rows: list[dict[str, Any]], output_dir: Path, batch_size: int) -> list[dict[str, Any]]:
    import torch
    from tqdm.auto import tqdm
    from gpu_runtime import configure_cuda
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
    root = Path("results/a100_macro_f1/search_greedy")
    adapters = {
        "finetuned_step100": root / "train_full_2epochs/checkpoint-100",
        "search_best_step150": root / "greedy_01/checkpoint-150",
    }
    for adapter in adapters.values():
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Adapter checkpoint is incomplete: {adapter}")
    run_keys = {
        name: json.dumps({"model_id": MODEL_ID, "checkpoint": "base" if name == "qwen_base" else str(adapter.resolve()),
                          "dataset_revision": REVISION, "prompt": PROMPT, "methods": METHODS}, sort_keys=True)
        for name, adapter in {"qwen_base": None, **adapters}.items()
    }
    model_names = tuple(run_keys)
    partials = {name: output_dir / f"{name}.partial.json" for name in model_names}
    predictions = {name: read_resume(partials[name], run_keys[name], rows) for name in model_names}
    print("Loading shared Qwen2.5-Omni base and two LoRA adapters; " + ", ".join(
        f"{name}={len(predictions[name])}/{len(rows)} journaled" for name in model_names), flush=True)
    configure_cuda()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Evaluation requires a CUDA GPU with BF16 support")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapters["finetuned_step100"], adapter_name="ft100")
    model.load_adapter(adapters["search_best_step150"], adapter_name="search150")
    model.eval()
    thinker = model.base_model.model.thinker
    tokenizer = processor.tokenizer
    yes_ids, no_ids = answer_token_ids(tokenizer)
    yes_encoding = tokenizer.encode(" yes", add_special_tokens=False)
    no_encoding = tokenizer.encode(" no", add_special_tokens=False)
    if len(yes_encoding) != 1 or len(no_encoding) != 1:
        raise ValueError(f"Single-token method requires one token each: ' yes'={yes_encoding}, ' no'={no_encoding}")
    single_yes, single_no = yes_encoding[0], no_encoding[0]
    if single_yes == single_no:
        raise ValueError("Single-token yes/no IDs overlap")
    print(f"TAD token-set sizes yes={len(yes_ids)} no={len(no_ids)}; "
          f"single IDs yes={single_yes}, no={single_no}; requested batch={batch_size}.", flush=True)
    checkpoint_paths = {"qwen_base": "base", **{name: str(path.resolve()) for name, path in adapters.items()}}
    metadata = {
        name: {"model_id": MODEL_ID, "checkpoint": checkpoint_paths[name], "dataset_revision": REVISION,
               "prompt_template": PROMPT,
               "system_prompt": "Listen to the audio and answer the question with exactly Yes or No.",
               "tad_yes_token_ids": yes_ids, "tad_no_token_ids": no_ids,
               "single_token_ids": {" yes": single_yes, " no": single_no},
               "requested_batch_size": batch_size}
        for name in model_names
    }
    for name in model_names:
        atomic_json(partials[name], {"run_key": run_keys[name], "metadata": metadata[name],
                                     "predictions": predictions[name]})
    index = 0
    completed = sum(all(row["id"] in predictions[name] for name in model_names) for row in rows)
    progress = tqdm(total=len(rows), initial=completed, desc="StemFoley (3 models)", unit="item", dynamic_ncols=True)
    while index < len(rows):
        while index < len(rows) and all(rows[index]["id"] in predictions[name] for name in model_names):
            index += 1
        if index == len(rows):
            break
        current_batch = min(batch_size, len(rows) - index)
        batch = rows[index:index + current_batch]
        try:
            inputs = batch_inputs(batch, processor, model.device, model.dtype)
            batch_predictions = {}
            with torch.inference_mode():
                with model.disable_adapter():
                    output = thinker(**inputs, use_cache=False)
                batch_predictions["qwen_base"] = token_scores(
                    output.logits[:, -1, :], yes_ids, no_ids, single_yes, single_no)
                del output
                for model_name, adapter_name in (("finetuned_step100", "ft100"),
                                                 ("search_best_step150", "search150")):
                    model.set_adapter(adapter_name)
                    output = thinker(**inputs, use_cache=False)
                    batch_predictions[model_name] = token_scores(
                        output.logits[:, -1, :], yes_ids, no_ids, single_yes, single_no)
                    del output
            for model_name, scores in batch_predictions.items():
                for row, score in zip(batch, scores):
                    predictions[model_name][row["id"]] = score
                atomic_json(partials[model_name], {"run_key": run_keys[model_name],
                                                   "metadata": metadata[model_name],
                                                   "predictions": predictions[model_name]})
            progress.update(len(batch))
            index += current_batch
            del inputs, batch_predictions
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or current_batch <= 1:
                progress.close()
                raise
            if "inputs" in locals():
                del inputs
            if "output" in locals():
                del output
            gc.collect()
            torch.cuda.empty_cache()
            batch_size = max(1, current_batch // 2)
            progress.set_postfix(batch_size=batch_size, status="OOM; reducing")
    progress.close()
    records = []
    for name in model_names:
        records.extend(write_model_outputs(output_dir, name, rows, predictions[name], metadata[name]))
        partials[name].unlink(missing_ok=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return records


def main() -> None:
    import logging
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stemfoley_eval"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    dataset_dir = args.dataset_dir or Path(snapshot_download(
        repo_id="snowchan/StemFoley", repo_type="dataset", revision=REVISION,
        allow_patterns=["test_labels.csv", "test_audio/**"]))
    rows = load_rows(dataset_dir)
    print(f"StemFoley rows={len(rows)}, in_final_v3={sum(r['in_final_v3'] for r in rows)}, "
          f"easy={sum(r['difficulty'] == 'easy' for r in rows)}, "
          f"hard={sum(r['difficulty'] == 'hard' for r in rows)}", flush=True)

    all_records = evaluate_all(rows, args.output_dir, args.batch_size)
    with (args.output_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "subset", "method", "n", "accuracy", "macro_f1", "yes_f1", "no_f1"]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"Completed StemFoley evaluation. Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
