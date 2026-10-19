"""Budgeted ms-swift jobs; JSON events go to Optuna and readable logs to the shell."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiment_utils import read_rows, write_json, overfit_verdict
from qwen_train import MODEL_ID, SYSTEM, evaluate
from search_policy import accumulation_for

EVENT_PREFIX = "AQA_EVENT "
TARGET_REGEX = (r".*thinker\.model\.layers\.\d+\."
                r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
                r"mlp\.(gate_proj|up_proj|down_proj))")


def swift_record(row: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "<audio>" + row["question"]},
        {"role": "assistant", "content": row["answer"].capitalize()},
    ], "audios": [row["audio_path"]]}


def thinker_of(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.thinker


def emit(kind: str, **values):
    print(EVENT_PREFIX + json.dumps({"kind": kind, **values}, allow_nan=False), flush=True)


def gpu_snapshot() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True)
        util, used, total = map(float, result.stdout.splitlines()[0].split(","))
        return {"gpu_utilization_percent": util, "gpu_memory_used_mib": used, "gpu_memory_total_mib": total}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def run_job(job: dict) -> dict:
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["USE_HF"] = "1"
    import torch
    from swift.llm import TrainArguments
    from swift.llm.train.sft import SwiftSft
    from transformers import TrainerCallback
    from gpu_runtime import configure_cuda

    configure_cuda()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    training = read_rows(Path(job["train_path"]), allow_repeated=True)
    validation = read_rows(Path(job["val_path"])) if job.get("val_path") else []
    dataset_path = output / "train_swift.jsonl"
    with dataset_path.open("w", encoding="utf-8") as stream:
        for row in training:
            stream.write(json.dumps(swift_record(row), ensure_ascii=False) + "\n")
    mode = job.get("mode", "search")
    max_steps = job["max_steps"]
    best = {"macro_f1": -1.0}
    result = {}
    history, losses, durations = [], [], []
    best_step, stale = 0, 0
    stop_reason = "max_steps"
    before = {}

    class MonitorCallback(TrainerCallback):
        def __init__(self, processor):
            self.processor = processor
            self.writer = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            if not names or any("lora_" not in n or "thinker.model.layers." not in n for n in names):
                raise RuntimeError(f"Unexpected trainable parameters: {names}")
            write_json(output / "trainable_parameters.json", names)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(output / "runs" / "metrics"))
            self.started = self.last_step_end = time.monotonic()
            if mode == "overfit":
                before.update(self.measure(model))
                self.started = self.last_step_end = time.monotonic()
            emit("started", mode=mode, max_steps=max_steps, batch_size=job["batch_size"],
                 accumulation=accumulation_for(job["batch_size"]))

        def measure(self, model, rows=None, output_path=None):
            try:
                return evaluate(thinker_of(model), self.processor,
                                validation if rows is None else rows, output_path)
            finally:
                self.processor.tokenizer.padding_side = "right"
                model.train()

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                if not math.isfinite(float(logs["loss"])):
                    raise RuntimeError("Nonfinite training loss")
                losses.append(float(logs["loss"]))
            elapsed = time.monotonic() - self.started
            seconds = statistics.median(durations[-20:]) if durations else 0
            record = {**logs, "step": state.global_step, "max_steps": max_steps,
                      "elapsed_seconds": elapsed, "seconds_per_step": seconds,
                      "remaining_train_seconds": max(0, max_steps-state.global_step) * seconds,
                      "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                      "reserved_gib": torch.cuda.memory_reserved() / 1024**3, **gpu_snapshot()}
            with (output / "step_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            write_json(output / "status.json", {"state": "RUNNING", "mode": mode, **record})
            for key, value in record.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.writer.add_scalar(f"progress/{key}", value, state.global_step)
            self.writer.flush()
            emit("progress", **record)

        def validate(self, model, state, control):
            nonlocal best_step, stale, stop_reason
            score = self.measure(model)
            step = state.global_step
            value = score["macro_f1"]
            if value is None or not math.isfinite(value):
                raise RuntimeError("Validation needs both labels and finite metrics")
            improved = value > best["macro_f1"]
            stale = 0 if improved else stale + 1
            if improved:
                best.clear()
                best.update(score)
                best_step = step
                model.save_pretrained(output / "best_adapter", safe_serialization=True)
                self.processor.save_pretrained(output / "best_adapter")
            history.append({"step": step, **score})
            for key, metric in score.items():
                if isinstance(metric, (int, float)):
                    self.writer.add_scalar(f"val/{key}", metric, step)
            self.writer.add_scalar("val/baseline_macro_f1", job["baseline"], step)
            self.writer.flush()
            write_json(output / "history.json", history)
            print(f"[{mode}] step {step}/{max_steps}: val macro_f1={value:.4f}, "
                  f"accuracy={score['accuracy']:.4f}, "
                  f"best={best['macro_f1']:.4f}, baseline={job['baseline']:.4f}", flush=True)
            emit("validation", step=step, score=value, best=best["macro_f1"], best_step=best_step)
            if job.get("interactive"):
                line = sys.stdin.readline()
                if not line:
                    raise RuntimeError("Search controller disconnected during validation")
                decision = json.loads(line)
                if decision.get("prune"):
                    stop_reason = decision.get("reason", "optuna_pruned")
                    control.should_training_stop = True
            if stale >= job.get("patience", 2) and step >= job.get("min_steps", 100):
                stop_reason = "no_improvement"
                control.should_training_stop = True
            self.last_step_end = time.monotonic()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            nonlocal stop_reason
            # Synchronize only benchmarks; normal training can overlap CPU/GPU work.
            if mode == "profile":
                torch.cuda.synchronize()
            now = time.monotonic()
            durations.append(now - self.last_step_end)
            self.last_step_end = now
            deadline_reached = job.get("deadline") is not None and time.time() >= job["deadline"]
            if mode == "search" and (state.global_step % job["eval_steps"] == 0
                                      or state.global_step == max_steps or deadline_reached):
                self.validate(model, state, control)
                if deadline_reached:
                    stop_reason = "budget_exhausted"
                    control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if mode == "overfit":
                original_path = output / 'tiny_predictions.tsv'
                after = self.measure(model, output_path=original_path)
                result.update(overfit_verdict(losses, before, after))
                if job.get('ablation_path'):
                    silence_path = output / 'silence_predictions.tsv'
                    muted_score = self.measure(model, read_rows(Path(job['ablation_path'])), silence_path)
                    def predictions(path):
                        with path.open(newline='') as stream:
                            return {r['id']: r['prediction'] for r in csv.DictReader(stream, delimiter='\t')}
                    normal, muted = predictions(original_path), predictions(silence_path)
                    result['audio_ablation'] = {
                        'silence_accuracy': muted_score['accuracy'],
                        'prediction_change_rate': sum(normal[k] != muted[k] for k in normal) / len(normal),
                        'note': 'Diagnostic only; low change can also reflect baseline answer bias.',
                    }
            seconds = statistics.median(durations[2:] or durations)
            result.update({"mode": mode, "completed_steps": state.global_step,
                           "seconds_per_step": seconds, "examples_per_second": 64 / seconds,
                           "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                           "batch_size": job["batch_size"], "workers": job["workers"],
                           "gradient_checkpointing": job["gradient_checkpointing"],
                           "stop_reason": stop_reason, "best_step": best_step,
                           "best": best if best_step else None,
                           "adapter": str((output / "best_adapter").resolve()) if best_step else None})
            self.writer.close()
            write_json(output / "result.json", result)
            write_json(output / "status.json", {"state": "FINISHED", **result})
            emit("finished", **result)

    class ClothoSft(SwiftSft):
        def _prepare_callbacks(self):
            super()._prepare_callbacks()
            self.callbacks.append(MonitorCallback(self.processor))

        def train(self, trainer):
            example = trainer.train_dataset[0]
            labels = example["labels"]
            supervised = [int(token) for token in labels if int(token) != -100]
            if not 1 <= len(supervised) <= 16 or not any(int(token) == -100 for token in labels):
                raise RuntimeError("Unexpected label mask; only the short assistant answer should be supervised")
            decoded = self.processor.tokenizer.decode(supervised, skip_special_tokens=True).strip()
            if decoded.lower() not in ('yes', 'no'):
                raise RuntimeError(f'Unexpected supervised answer: {decoded!r}')
            write_json(output / "label_check.json", {
                "supervised_tokens": supervised,
                "text": decoded,
                "masked_tokens": sum(int(token) == -100 for token in labels),
            })
            return super().train(trainer)

    workers = job["workers"]
    args = TrainArguments(
        model=MODEL_ID, model_type="qwen2_5_omni", train_type="lora",
        dataset=[str(dataset_path.resolve())], split_dataset_ratio=0,
        torch_dtype="bfloat16", bf16=True, tf32=True, attn_impl="sdpa",
        lora_rank=job["rank"], lora_alpha=16, lora_dropout=0.05,
        target_regex=TARGET_REGEX, freeze_vit=True, freeze_aligner=True,
        max_steps=max_steps, num_train_epochs=1, learning_rate=job["learning_rate"], weight_decay=0.1,
        adam_beta1=0.9, adam_beta2=0.999,
        per_device_train_batch_size=job["batch_size"],
        gradient_accumulation_steps=accumulation_for(job["batch_size"]),
        gradient_checkpointing=job["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=min(10, max_steps // 10), warmup_ratio=0, lr_scheduler_type="linear", optim="adamw_torch",
        eval_strategy="no", save_strategy="no", report_to=["tensorboard"],
        logging_strategy="steps", logging_steps=1 if mode != "search" else 10, logging_first_step=True,
        logging_dir=str((output / "runs").resolve()),
        output_dir=str(output.resolve()), add_version=False,
        max_length=8192, truncation_strategy="raise", strict=True,
        lazy_tokenize=True, dataset_num_proc=1, dataset_shuffle=False, train_dataloader_shuffle=False,
        dataloader_num_workers=workers, dataloader_persistent_workers=workers > 0,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if workers else None,
        seed=job["seed"], data_seed=job["seed"],
    )
    ClothoSft(args).main()
    return result


def evaluate_adapter(adapter: Path, data_dir: Path, output_dir: Path, split: str = "test") -> dict:
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["USE_HF"] = "1"
    import torch
    from swift.llm import get_model_tokenizer
    from peft import PeftModel
    from gpu_runtime import configure_cuda
    configure_cuda()
    model, processor = get_model_tokenizer(
        MODEL_ID, model_type="qwen2_5_omni", torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_impl="sdpa")
    model = PeftModel.from_pretrained(model, str(adapter))
    output_dir.mkdir(parents=True, exist_ok=True)
    return evaluate(thinker_of(model), processor, read_rows(data_dir / f"{split}.jsonl"),
                    output_dir / f"{split}_predictions.tsv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    try:
        if args.job:
            run_job(json.loads(args.job.read_text()))
        elif args.adapter and args.data_dir and args.output_dir:
            score = evaluate_adapter(args.adapter, args.data_dir, args.output_dir, args.split)
            write_json(args.output_dir / "evaluation.json", score)
        else:
            parser.error("Use --job or --adapter/--data-dir/--output-dir")
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            if args.job:
                job = json.loads(args.job.read_text())
                write_json(Path(job['output_dir']) / 'status.json', {'state': 'FAILED', 'error': str(error)})
            raise
        if args.job:
            job = json.loads(args.job.read_text())
            write_json(Path(job['output_dir']) / 'status.json', {'state': 'OOM', 'error': str(error)})
        print(f"CUDA OOM: {error}", flush=True)
        raise SystemExit(75)


if __name__ == "__main__":
    main()
