"""Qwen2.5-Omni Thinker LoRA training and deterministic Yes/No evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import random
from pathlib import Path
from typing import Callable
from functools import lru_cache

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SYSTEM = "Listen to the audio and answer the question with exactly Yes or No."


@lru_cache(maxsize=128)
def evaluation_audio(path: str, sample_rate: int):
    """Bounded CPU waveform cache shared by repeated validation questions."""
    import librosa
    return librosa.load(path, sr=sample_rate, mono=True)[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    for row in rows:
        if row["answer"] not in {"yes", "no"} or not Path(row["audio_path"]).is_file():
            raise ValueError(f"Invalid answer or missing audio: {row}")
    return rows


def messages(question: str, answer: str | None = None) -> list[dict]:
    result = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": "audio.wav"},
            {"type": "text", "text": question},
        ]},
    ]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result


def batch_for(row: dict, processor, device, dtype, train: bool):
    import librosa
    import torch

    sample_rate = processor.feature_extractor.sampling_rate
    waveform, _ = librosa.load(row["audio_path"], sr=sample_rate, mono=True)
    prompt = processor.apply_chat_template(messages(row["question"]), tokenize=False,
                                           add_generation_prompt=True)
    def encode(text):
        encoded = processor(text=[text], audio=[waveform], sampling_rate=sample_rate,
                            padding=True, return_tensors="pt")
        return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                for key, value in encoded.items()}
    if not train:
        return encode(prompt)
    full = processor.apply_chat_template(messages(row["question"], row["answer"].capitalize()),
                                         tokenize=False, add_generation_prompt=False)
    prompt_inputs, full_inputs = encode(prompt), encode(full)
    prefix = prompt_inputs["input_ids"][0]
    sequence = full_inputs["input_ids"][0]
    if len(sequence) <= len(prefix) or not torch.equal(sequence[:len(prefix)], prefix):
        raise ValueError("Assistant answer is not a token prefix continuation; inspect chat template")
    labels = sequence.clone()
    labels[:len(prefix)] = -100
    full_inputs["labels"] = labels.unsqueeze(0)
    return full_inputs


def batch_for_evaluation(rows: list[dict], processor, device, dtype):
    import torch

    sample_rate = processor.feature_extractor.sampling_rate
    texts, waveforms = [], []
    for row in rows:
        waveform = evaluation_audio(row["audio_path"], sample_rate)
        texts.append(processor.apply_chat_template(
            messages(row["question"]), tokenize=False, add_generation_prompt=True))
        waveforms.append(waveform)
    encoded = processor(text=texts, audio=waveforms, sampling_rate=sample_rate,
                        padding=True, return_tensors="pt")
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in encoded.items()}


def batch_for_training(rows: list[dict], processor, device, dtype):
    import librosa
    import torch

    # Evaluation uses left padding for generation; causal-LM labels need right padding.
    processor.tokenizer.padding_side = "right"
    sample_rate = processor.feature_extractor.sampling_rate
    waveforms = [librosa.load(row["audio_path"], sr=sample_rate, mono=True)[0] for row in rows]
    prompts = [processor.apply_chat_template(messages(row["question"]), tokenize=False,
                                              add_generation_prompt=True) for row in rows]
    full_texts = [processor.apply_chat_template(
        messages(row["question"], row["answer"].capitalize()), tokenize=False,
        add_generation_prompt=False) for row in rows]
    prompt_inputs = processor(text=prompts, audio=waveforms, sampling_rate=sample_rate,
                              padding=True, return_tensors="pt")
    full_inputs = processor(text=full_texts, audio=waveforms, sampling_rate=sample_rate,
                            padding=True, return_tensors="pt")
    labels = full_inputs["input_ids"].clone()
    for index in range(len(rows)):
        prompt_length = int(prompt_inputs["attention_mask"][index].sum())
        full_length = int(full_inputs["attention_mask"][index].sum())
        if full_length <= prompt_length or not torch.equal(
                full_inputs["input_ids"][index, :prompt_length],
                prompt_inputs["input_ids"][index, :prompt_length]):
            raise ValueError("Assistant answer is not a token prefix continuation; inspect chat template")
        labels[index, :prompt_length] = -100
        labels[index, full_length:] = -100
    full_inputs["labels"] = labels
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in full_inputs.items()}


def select_training_batch_size(training, processor, model, requested: int) -> int:
    import gc
    import soundfile as sf
    import torch

    durations = [(sf.info(row["audio_path"]).duration, index) for index, row in enumerate(training)]
    longest = [training[index] for _, index in sorted(durations, reverse=True)[:requested]]
    candidates = [size for size in (64, 32, 16, 8, 4, 2, 1) if size <= requested]
    model.train()
    for size in candidates:
        try:
            inputs = batch_for_training(longest[:size], processor, model.device, model.dtype)
            model.zero_grad(set_to_none=True)
            model(**inputs, use_cache=False).loss.backward()
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
            return size
        except RuntimeError as error:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if "out of memory" not in str(error).lower() or size == 1:
                raise
    raise RuntimeError("Could not find a training microbatch size")


def normalize_answer(value: str) -> str | None:
    answer = value.strip().lower().strip(" .!\n\t")
    return answer if answer in {"yes", "no"} else None


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["answer"] == r["prediction"] for r in rows)
    f1s = []
    for label in ("yes", "no"):
        tp = sum(r["answer"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["answer"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["answer"] == label and r["prediction"] != label for r in rows)
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return {"n": n, "accuracy": correct / n, "macro_f1": sum(f1s) / 2,
            "invalid_outputs": sum(r["prediction"] is None for r in rows)}


def evaluate(model, processor, rows: list[dict], output_path: Path | None = None) -> dict:
    import torch
    from tqdm.auto import tqdm

    model.eval()
    results = []
    batch_size = 32
    processor.tokenizer.padding_side = "left"
    progress = tqdm(total=len(rows), desc="Evaluation", unit="item",
                    dynamic_ncols=True, leave=True)
    start = 0
    while start < len(rows):
        batch = rows[start:start + batch_size]
        try:
            inputs = batch_for_evaluation(batch, processor, model.device, model.dtype)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                           pad_token_id=processor.tokenizer.pad_token_id)
            new_ids = generated[:, inputs["input_ids"].shape[1]:]
            decoded = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            results.extend({"id": row["id"], "answer": row["answer"],
                            "prediction": normalize_answer(raw), "raw_output": raw}
                           for row, raw in zip(batch, decoded))
            start += len(batch)
            progress.update(len(batch))
            del inputs, generated
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            if "inputs" in locals():
                del inputs
            if "generated" in locals():
                del generated
            torch.cuda.empty_cache()
            progress.set_postfix(batch_size=batch_size, status="OOM; reducing")
    progress.close()
    score = metrics(results)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "answer", "prediction", "raw_output"],
                                    delimiter="\t")
            writer.writeheader()
            writer.writerows(results)
        output_path.with_suffix(".metrics.json").write_text(json.dumps(score, indent=2) + "\n")
    return score


def load_model(rank: int | None = None, dropout: float = 0.0, adapter: Path | None = None):
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from gpu_runtime import configure_cuda
    configure_cuda()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training/evaluation requires a CUDA GPU with BF16 support")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    if rank is not None:
        from peft import LoraConfig, get_peft_model
        for parameter in model.audio_tower.parameters():
            parameter.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=16, lora_dropout=dropout, bias="none",
            target_modules=(r".*model\.layers\.\d+\."
                            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
                            r"mlp\.(gate_proj|up_proj|down_proj))"),
        ))
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("LoRA matched no trainable parameters")
        if any("audio_tower" in name for name in trainable):
            raise RuntimeError("Audio encoder must remain frozen")
        if any("model.layers." not in name for name in trainable):
            raise RuntimeError("Only Thinker text transformer layers may be trainable")
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, processor


def train(data_dir: Path, output_dir: Path, *, learning_rate: float = 5e-5,
          rank: int = 8, dropout: float = 0.05, epochs: int = 3,
          batch_size: int = 64, effective_batch_size: int = 64,
          seed: int = 42, save_adapter: bool = True,
          on_epoch: Callable[[int, dict], None] | None = None) -> dict:
    import torch
    from tqdm.auto import tqdm

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    training = read_jsonl(data_dir / "train.jsonl")
    validation = read_jsonl(data_dir / "val.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(rank, dropout)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    processor.tokenizer.padding_side = "right"
    batch_size = select_training_batch_size(training, processor, model, batch_size)
    accumulation = max(1, math.ceil(effective_batch_size / batch_size))
    print(f"Selected training microbatch={batch_size}, gradient accumulation={accumulation}, "
          f"effective batch≈{batch_size * accumulation}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)
    microbatches_per_epoch = math.ceil(len(training) / batch_size)
    steps_per_epoch = math.ceil(microbatches_per_epoch / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, math.ceil(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup,
                                    max(0.0, (total_steps - step) / max(1, total_steps - warmup))))
    best = {"macro_f1": -1.0, "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(training)))
        random.Random(seed + epoch).shuffle(order)
        batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(total=len(batches), desc=f"Epoch {epoch}/{epochs}", unit="batch",
                        dynamic_ncols=True, leave=True)
        for group_start in range(0, len(batches), accumulation):
            group = batches[group_start:group_start + accumulation]
            sample_count = sum(len(batch) for batch in group)
            for positions in group:
                inputs = batch_for_training([training[position] for position in positions],
                                            processor, model.device, model.dtype)
                loss = model(**inputs, use_cache=False).loss * (len(positions) / sample_count)
                del inputs
                loss.backward()
                progress.update(1)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(lr=f"{scheduler.get_last_lr()[0]:.2g}")
        progress.close()
        score = evaluate(model, processor, validation)
        record = {"epoch": epoch, **score}
        history.append(record)
        print(f"Epoch {epoch}/{epochs} validation: macro_f1="
              f"{score['macro_f1']:.4f}, accuracy={score['accuracy']:.4f}", flush=True)
        if score["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, **score}
            if save_adapter:
                model.save_pretrained(output_dir / "best_adapter")
                processor.save_pretrained(output_dir / "best_adapter")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if on_epoch:
            on_epoch(epoch, record)
    (output_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "train", "test"], required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "train":
        result = train(args.data_dir, args.output_dir, learning_rate=args.learning_rate,
                       rank=args.rank, dropout=args.dropout, epochs=args.epochs,
                       batch_size=args.batch_size, effective_batch_size=args.effective_batch_size,
                       seed=args.seed)
    else:
        if args.mode == "test" and args.adapter is None:
            parser.error("--adapter is required for test mode")
        model, processor = load_model(adapter=args.adapter)
        split = "val" if args.mode == "baseline" else "test"
        result = evaluate(model, processor, read_jsonl(args.data_dir / f"{split}.jsonl"),
                          args.output_dir / f"{split}_predictions.tsv")
        result["dataset_sha256"] = hashlib.sha256((args.data_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        (args.output_dir / f"{split}_predictions.metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
