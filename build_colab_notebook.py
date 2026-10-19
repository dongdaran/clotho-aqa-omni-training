"""Embed the current training scripts into the Colab notebook for SSH workspaces."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ("clotho_data.py", "audio_augmentation.py", "qwen_train.py", "swift_train.py",
           "search_policy.py", "experiment_utils.py", "baseline_utils.py", "monitoring.py",
           "tune.py", "finalize.py", "gpu_runtime.py", "requirements.txt")


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def main() -> None:
    notebook_path = ROOT / "run_on_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    # Remove a redundant mount cell added by the extension after the bundled setup cell.
    notebook["cells"] = [
        cell for index, cell in enumerate(notebook["cells"])
        if not (cell["cell_type"] == "code" and
                "".join(cell["source"]).strip() ==
                "from google.colab import drive\ndrive.mount('/content/drive')")
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, arcname=name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    notebook["cells"][0]["source"] = lines(
        "# Clotho-AQA Yes/No · Colab 실행\n\n"
        "이 노트북은 **VS Code의 Google Colab 커널**에서 실행합니다. 로컬 코드를 고치는 작업은 VS Code에서, GPU 실행은 Colab에서 합니다. "
        "모델은 공식 Qwen2.5-Omni-3B입니다. train/val/test 모두 질문 시작 단어에 관계없이 Yes/No 문항만 사용합니다. 학습 loss는 기존 답변 토큰 cross entropy를 유지합니다. 학습은 batch 64부터 GPU 메모리를 점검하고, 부족하면 32→16→8→4→2→1로 자동 낮춥니다. "
        "배치를 낮추면 gradient accumulation으로 effective batch 약 64를 유지합니다.\n\n"
        "1. 오른쪽 위 **Select Kernel → Colab → New Colab Server**에서 GPU 런타임을 고릅니다.\n"
        "2. GPU 셀을 실행해 VRAM을 확인합니다. BF16과 약 40GB 이상의 GPU가 필요하며, 80GB를 권장합니다.\n"
        "3. 아래 셀을 순서대로 실행합니다. Drive 마운트 코드와 학습 코드가 노트북 안에 포함되어 있어 별도 업로드가 필요 없습니다.\n\n"
        "Colab 세션이 끝나면 `/content` 파일이 사라집니다. 공식 데이터 ZIP과 결과 TSV·어댑터는 Drive에 저장합니다. "
        "학습 `.py` 파일을 수정한 뒤에는 로컬에서 `python3 build_colab_notebook.py`를 실행해 노트북을 갱신하세요.\n")
    notebook["cells"][2]["source"] = lines(
        "## Drive 마운트와 코드 준비\n\n"
        "다음 셀 맨 위에서 Google Drive를 마운트하고 현재 학습 코드를 `/content/training`에 풉니다. "
        "SSH 폴더에는 확장의 **Upload to Colab** 메뉴가 나타나지 않아도 됩니다.\n")
    notebook["cells"][3]["source"] = lines(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n\n"
        "import base64, io, zipfile\n"
        "from pathlib import Path\n"
        f"_project_bundle = {encoded!r}\n"
        "PROJECT = Path('/content/training')\n"
        "PROJECT.mkdir(parents=True, exist_ok=True)\n"
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(_project_bundle))) as _archive:\n"
        "    _archive.extractall(PROJECT)\n"
        "assert (PROJECT / 'qwen_train.py').is_file()\n"
        "assert Path('/content/drive/MyDrive').is_dir(), 'Google Drive 마운트를 확인하세요'\n"
        "DRIVE = Path('/content/drive/MyDrive/clotho_aqa_omni')\n"
        "DRIVE.mkdir(parents=True, exist_ok=True)\n"
        "print('Code:', PROJECT, 'Persistent storage:', DRIVE)\n")
    data_cell = "".join(notebook["cells"][6]["source"])
    old = ("subprocess.run(['python', str(PROJECT/'clotho_data.py'), '--csv-dir', str(SOURCE),\n"
           "                '--audio-dir', str(AUDIO), '--output-dir', str(PROJECT/'data/processed')], check=True)")
    new = ("_prepared = subprocess.run(['python', str(PROJECT/'clotho_data.py'), '--csv-dir', str(SOURCE),\n"
           "                '--audio-dir', str(AUDIO), '--output-dir', str(PROJECT/'data/processed')],\n"
           "                capture_output=True, text=True)\n"
           "print(_prepared.stdout)\n"
           "if _prepared.returncode:\n"
           "    raise RuntimeError('Clotho 데이터 준비 실패:\\n' + _prepared.stderr)")
    if old in data_cell:
        data_cell = data_cell.replace(old, new)
    elif new not in data_cell:
        raise ValueError("Could not find data preparation command in notebook")
    notebook["cells"][6]["source"] = lines(data_cell)
    notebook["cells"][5]["source"] = lines(
        "## 공식 데이터 준비와 train 증강\n\n"
        "공식 Clotho-AQA 데이터를 내려받고 train/val/test 모두 질문 시작 단어에 관계없이 Yes/No 문항만 남깁니다. train에 −30/−20/−10/+10/+20 dB 변형을 추가합니다. "
        "RMS 하한은 −65 dBFS이며, ±1 초과값도 클리핑 없이 32-bit float WAV에 저장합니다. 각 원본 문항에 무음/No도 하나씩 추가해 총 7배로 만듭니다. "
        "val/test는 원본을 유지합니다. ZIP과 CSV는 Drive에 보관하며 증강 WAV는 `/content/training/data/processed/augmented_audio/train`에 생성합니다.\n")
    notebook["cells"][7]["source"] = lines(
        "## ms-swift + Optuna: 짧은 실험부터\n\n"
        "A100 80GB 1장, BF16. 먼저 cell 8에서 원본 val baseline을 다시 측정하세요. "
        "① 데이터/분할/라벨 점검 ② OOM·데이터 로더·checkpointing 속도 측정 "
        "③ 32문항 과적합/라벨 마스크/무음 반응 검사 ④ 원본 문항마다 증강 7종 중 하나씩 추출 "
        "⑤ 50 step마다 고정된 val 200문항 평가 ⑥ rank 16에서 LR 우선 Optuna 탐색 "
        "⑦ 쉘·TensorBoard·Optuna Dashboard 모니터링 순서입니다.\n\n"
        "LR 5e-5와 1e-4를 먼저 비교하고 세 번째 LR은 TPE가 고릅니다. 유망한 결과가 있으면 해당 LR로 rank 32를 한 번 더 비교합니다. "
        "각 trial은 최대 150 optimizer step, warmup 최대 10 step, 100 step부터 MedianPruner가 낮은 성능의 trial을 중단합니다. "
        "alpha 16, dropout 0.05, weight decay 0.1, effective batch 64를 유지합니다. 전체 예산은 준비 작업 포함 약 6시간이며 "
        "검증과 마무리는 예산을 넘길 수 있습니다. 최종 후보 최대 2개는 전체 val로 다시 비교합니다. "
        "audio encoder와 원본 가중치는 고정하며 Thinker 텍스트 attention/MLP LoRA만 학습합니다. "
        "새 결과는 `results/optuna_budgeted/`에 저장하고 기존 grid 결과는 보존합니다.\n")
    notebook["cells"][7]["source"] += lines(
        "\n탐색 셀은 학습 시작 전에 TensorBoard와 Optuna Dashboard 링크를 표시합니다. Scalars에서 `train/loss`, `train/learning_rate`, `train/grad_norm`을 "
        "첫 step과 이후 10 optimizer step마다 확인할 수 있습니다. val accuracy·macro-F1은 50 step마다 기록합니다. "
        "각 조합과 OOM batch 시도는 별도 run으로 표시됩니다. 그래프는 새로고침하거나 자동 갱신을 사용하세요. "
        "원시 step 기록은 각 batch 폴더의 `step_metrics.jsonl`에 저장합니다.\n")
    notebook["cells"][10]["source"] = lines(
        "## 최적 체크포인트의 test 평가\n\n"
        "짧은 탐색과 전체 val 재평가가 끝나면 전체 val 최고 점수의 어댑터로 test를 한 번 평가합니다. "
        "탐색에서 저장한 어댑터를 사용하며 별도로 재학습하지 않습니다. 결과는 `results/optuna_final/`에 저장됩니다.\n")
    notebook["cells"][4]["source"] = lines(
        "import subprocess, runpy, sys\n"
        "# Colab may include an old optional torchao; PEFT rejects it even though this run uses plain LoRA.\n"
        "subprocess.run(['python', '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)\n"
        "subprocess.run(['python', '-m', 'pip', 'install', '-r', str(PROJECT / 'requirements.txt')], check=True)\n"
        "subprocess.run(['python', '-c', 'import importlib.util, torch, transformers, peft, swift, librosa, tqdm; assert importlib.util.find_spec(\"torchao\") is None, \"Remove the incompatible optional torchao package\"; print(torch.__version__, transformers.__version__, peft.__version__)'], check=True)\n"
        "if str(PROJECT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT))\n"
        "def run_script(script, *args):\n"
        "    old_argv = sys.argv\n"
        "    sys.argv = [str(PROJECT / script), *args]\n"
        "    try:\n"
        "        runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "    finally:\n"
        "        sys.argv = old_argv\n"
        "        import gc, torch\n"
        "        gc.collect()\n"
        "        torch.cuda.empty_cache()\n")
    notebook["cells"][8]["source"] = lines(
        "run_script('qwen_train.py', '--mode', 'baseline',\n"
        "           '--data-dir', str(PROJECT/'data/processed'),\n"
        "           '--output-dir', str(DRIVE/'results/baseline_3b'))\n")
    notebook["cells"][9]["source"] = lines(
        "from monitoring import start_monitoring\n"
        "SEARCH_ROOT = DRIVE / 'results/optuna_budgeted'\n"
        "monitor_links = start_monitoring(SEARCH_ROOT)\n\n"
        "assert (DRIVE/'results/baseline_3b/val_predictions.metrics.json').is_file(), \\\n"
        "       '먼저 cell 8에서 3B baseline을 완료하세요'\n"
        "run_script('tune.py', '--data-dir', str(PROJECT/'data/processed'),\n"
        "           '--baseline-metrics', str(DRIVE/'results/baseline_3b/val_predictions.metrics.json'),\n"
        "           '--output-dir', str(SEARCH_ROOT), '--trials', '3', '--max-steps', '150',\n"
        "           '--eval-steps', '50', '--min-steps', '100', '--timeout-seconds', '21600')\n")
    notebook["cells"][11]["source"] = lines(
        "run_script('finalize.py', '--data-dir', str(PROJECT/'data/processed'),\n"
        "           '--best-tsv', str(DRIVE/'results/optuna_budgeted/best.tsv'),\n"
        "           '--output-dir', str(DRIVE/'results/optuna_final'))\n")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # A second CPU runtime can read Drive logs while the training kernel is busy.
    live_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": lines(
            "# 실행 중인 학습 로그 모니터링\n\n"
            "학습 노트북은 그대로 두고 이 노트북을 **별도 Colab CPU 런타임**에서 실행하세요. "
            "Drive의 기존 ms-swift logging.jsonl을 읽어 TensorBoard 그래프로 표시합니다. "
            "학습 프로세스나 원본 로그는 수정하지 않습니다. 로그가 아직 없으면 첫 기록까지 기다립니다. "
            "Drive 동기화로 표시가 지연될 수 있습니다.\n")},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "import base64, io, zipfile, subprocess, sys\n"
            "from pathlib import Path\n"
            f"_bundle = {encoded!r}\n"
            "PROJECT = Path('/content/aqa_monitor')\n"
            "PROJECT.mkdir(parents=True, exist_ok=True)\n"
            "with zipfile.ZipFile(io.BytesIO(base64.b64decode(_bundle))) as archive:\n"
            "    archive.extractall(PROJECT)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorboard>=2.18,<3'], check=True)\n"
            "sys.path.insert(0, str(PROJECT))\n"
            "from monitoring import start_monitoring\n"
            "ROOT = Path('/content/drive/MyDrive/clotho_aqa_omni/results/swift_aug_grid')\n"
            "links = start_monitoring(ROOT, legacy=True)\n")},
    ]
    live = {"cells": live_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 4}
    (ROOT / 'monitor_existing.ipynb').write_text(json.dumps(live, ensure_ascii=False, indent=1) + '\n')


if __name__ == "__main__":
    main()
