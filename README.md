## 현재 서버 실행 설정

현재 기본 서버 실행은 Optuna 없이 LR 후보 5e-5, 1e-4, 2.5e-5를 비교합니다. rank 16, microbatch 16, gradient accumulation 4를 고정합니다. 탐색은 같은 샘플링 입력을 후보마다 사용하며 후보당 최대 150 optimizer step, 50 step마다 val 200문항을 평가합니다. 작은 val 상위 최대 2개를 전체 val로 재평가해 선택합니다. 이전 Optuna 실행 설명은 아래에 기록으로 남아 있습니다.

최종 학습은 원본 4,695문항과 증강 28,170행을 모두 포함한 32,865행이 1 epoch입니다. epoch마다 전체 행을 seed 42로 독립적으로 섞고, 2 epoch(65,730행, 약 1,028 optimizer step)를 학습합니다. 전체 val을 100 step마다와 epoch 경계에서 평가합니다. 최고 macro-F1 가중치는 `best_adapter/`, 최고 재개용 체크포인트 경로와 점수는 `best_checkpoint.json`에 저장합니다. 평가 시점과 3,000 step마다 재개용 체크포인트를 저장하며, 종료 가중치는 `final_adapter/`에 저장합니다.

결과는 `results/a100_macro_f1/search_greedy/`에 저장합니다. 탐색 표는 `trials.tsv`, 선택 설정은 `best.tsv`, 최종 학습은 `train_full_2epochs/`입니다. `trials.tsv`의 `sample_val_macro_f1`은 200문항 최고 점수, `full_val_macro_f1`은 최종 후보의 전체 val 점수입니다. Optuna pruning은 사용하지 않으며 작은 val 점수가 연속 개선되지 않으면 100 step 이후 후보 학습이 종료될 수 있습니다.

```bash
bash run_on_server.sh --stage search
bash run_on_server.sh --stage train
tail -f results/a100_macro_f1/pipeline.log
```

---

# Clotho-AQA Yes/No · Qwen2.5-Omni

**A100 80GB 일반 서버:** [서버 실행 안내](SERVER.md)를 따르세요. `bash run_on_server.sh`로 데이터 준비부터 baseline·Optuna 탐색·test까지 실행합니다. 일반 AdamW를 사용하고 accuracy·macro-F1을 기록하며 모델 선택은 macro-F1을 기준으로 합니다.

이 폴더에는 실행 코드만 있습니다. 데이터 다운로드, 모델 로딩, ms-swift 학습과 탐색은 **GPU 서버에서** 실행하세요. 공식 `Qwen/Qwen2.5-Omni-3B` 체크포인트를 사용합니다. train/val/test 모두 질문의 시작 단어는 제한하지 않습니다. 정답 주석이 `yes` 또는 `no`인 모든 문항을 사용합니다. 같은 오디오·질문에 대한 세 주석은 다수결로 합칩니다.

## VS Code에서 Colab으로 실행

[run_on_colab.ipynb](run_on_colab.ipynb)을 VS Code에서 열고 **Select Kernel → Colab → New Colab Server**로 GPU 서버를 선택하세요. 노트북 첫 셀에서 GPU와 VRAM을 확인한 뒤 셀을 순서대로 실행하면 됩니다. 노트북에 Google Drive 마운트 코드와 학습 코드가 포함되어 있으므로 SSH 작업 폴더를 업로드할 필요가 없습니다. 학습 `.py` 파일을 수정한 뒤에는 로컬에서 `python3 build_colab_notebook.py`로 노트북을 갱신하세요. 연결 메뉴는 [Google Colab VS Code 확장 사용 안내](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)를 따릅니다.

노트북은 공식 데이터만 다운로드하고, 탐색 TSV·최적 어댑터·평가 결과를 Drive의 `MyDrive/clotho_aqa_omni/`에 저장합니다. Colab의 GPU 종류와 실행 시간은 고정되어 있지 않으므로 80GB VRAM을 권장하며, 40GB급 GPU에서는 메모리 부족 가능성이 있습니다. 아래 명령줄 절차는 다른 GPU 서버에서도 사용할 수 있습니다.

## 권장 서버와 설치

- 시작 권장 사양: BF16을 지원하는 **80GB VRAM GPU 1장**, 여유 디스크 30GB 이상. 80GB도 실제 메모리 적합성을 보장하지 않으므로 첫 epoch를 확인하세요.
- Python 3.12, CUDA에 맞는 PyTorch, `ffmpeg`가 필요합니다. PyTorch 설치 명령은 [공식 선택기](https://pytorch.org/get-started/locally/)에서 서버 CUDA 버전에 맞춰 고르세요.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# 먼저 서버 CUDA 버전에 맞는 PyTorch를 설치합니다.
# 사전 설치된 오래된 torchao가 있으면 PEFT가 LoRA 초기화에서 실패하므로 제거합니다.
.venv/bin/python -m pip uninstall -y torchao
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
```

## 공식 데이터

[Clotho-AQA 공식 배포본](https://zenodo.org/records/6473207)의 `audio_files.zip`과 train·val·test CSV를 사용합니다. 오디오는 `data/audio/`에 파일명 그대로 풀어야 합니다. 내려받을 때 원본 파일명을 보존하세요. `clotho_data.py`는 공식 CSV의 MD5, 오디오 존재 여부, 분할 간 오디오 중복을 검사합니다.

```bash
mkdir -p data/source data/audio
for split in train val test; do
  curl -fL "https://zenodo.org/records/6473207/files/clotho_aqa_${split}.csv?download=1" \
    -o "data/source/clotho_aqa_${split}.csv"
done
curl -fL 'https://zenodo.org/records/6473207/files/audio_files.zip?download=1' \
  -o data/source/audio_files.zip
echo '48c2c46ccd71dc06121228e2ed68d6b9  data/source/audio_files.zip' | md5sum -c -
unzip -j data/source/audio_files.zip -d data/audio
.venv/bin/python clotho_data.py --csv-dir data/source --audio-dir data/audio \
  --output-dir data/processed
```

이미 공식 train·test CSV가 있으면 그대로 사용할 수 있습니다. 현재 로컬 train·test 파일은 공식 MD5와 일치합니다. `--skip-md5`는 공식 CSV가 아닌 변형본을 의도적으로 사용할 때만 사용하세요.

데이터 준비 시 train의 각 원본 문항에 **−30, −20, −10, +10, +20 dB** 변형 5개와 **무음/No** 1개를 추가합니다(원본 포함 문항 수 7배). 음량 변형은 원래 답을 유지하며, RMS 기준 −65 dBFS 아래로 내려가지 않도록 감쇠량을 제한합니다. 원본이 이미 하한보다 작으면 하한까지 올리고, 원본이 완전 무음이면 그대로 둡니다. +10/+20 dB 적용 시 ±1을 넘는 값도 클리핑 없이 32-bit float WAV에 그대로 저장합니다. JSONL의 `samples_above_full_scale`에는 ±1 초과 샘플 수를 기록합니다. 무음은 하한의 예외로 디지털 0이며 원본과 길이·샘플레이트가 같습니다. 생성 오디오는 `data/processed/augmented_audio/train/`에 저장하고 같은 오디오의 여러 질문이 공유합니다. val/test는 원본을 유지합니다. 증강을 끄려면 `--no-augmentation`을 지정하세요. Colab에서도 데이터 준비 셀을 실행하면 동일하게 생성됩니다.

## 실험 순서: 작은 검증부터 Optuna 탐색까지

A100 80GB 1장, BF16, ms-swift 3.12.0을 사용합니다. 이 API와 맞추기 위해 Transformers 4.57.1을 고정합니다. 기존 환경에서 requirements를 갱신했다면 Colab 커널을 한 번 재시작한 뒤 코드 준비부터 다시 실행하세요. 원본 가중치와 audio encoder는 고정하고, Thinker 텍스트 attention(q/k/v/o)·MLP(gate/up/down)의 LoRA만 학습합니다. 실행 시 대상 이름과 실제 답변 토큰 마스크를 검사합니다.

기본 흐름은 다음 7단계입니다. 탐색은 기존 9조합×5 epoch 대신 step 예산을 사용합니다.

1. **데이터 점검:** Yes/No 라벨, 각 증강의 원본 연결과 라벨, 오디오 존재·길이·샘플레이트, train/val/test 간 원본 오디오 중복을 검사합니다. `data_audit.json`에 분포와 데이터 지문을 저장합니다. test는 이 단계에서 데이터 무결성만 확인하며 모델 선택 점수에는 사용하지 않습니다.
2. **속도/메모리 측정:** 긴 원본 오디오로 GPU batch 64부터 시도하고 OOM이면 32→16→8→4→2→1로 줄입니다. 성공 batch에서 데이터 로더 workers 0/2와 gradient checkpointing 켜기/끄기를 10 optimizer step씩 비교합니다. 성공한 설정 중 중앙 step 시간이 가장 작은 것을 선택하고 `performance.json`에 측정치와 예상 시간을 남깁니다. 선택적 성능 프로파일이 실패하면 정상 동작한 설정을 유지합니다.
3. **32문항 학습 검사:** 원본 train에서 Yes/No를 나눠 32문항을 뽑아 LR 1e-4로 30 step 반복합니다. 실제 라벨 마스크, LoRA 학습 대상, loss 감소와 학습 문항 정확도를 확인합니다. 마지막 loss가 처음보다 5% 이상 낮고 정확도가 유지되거나, 이미 정확도 95% 이상이며 loss가 크게 나빠지지 않으면 통과합니다. 원본과 무음의 예측 변화율도 진단용으로 저장합니다. 기준을 통과하지 못하면 탐색 전에 멈추고 결과를 확인하도록 합니다.
4. **원본별 증강 샘플링:** 원본 문항을 한 번씩 섞어서 순회하고, 각 방문마다 원본·음량 5종·무음 중 하나를 무작위로 고릅니다. 7종을 모두 한 epoch에 넣지 않습니다. 원본 방문 순서와 변형 선택은 seed로 고정해 모든 trial이 같은 입력 예산을 받습니다. 무음은 약 1/7, Yes/No 비율은 기존 증강 구성과 기대상 동일합니다. 저장된 7배 train JSONL과 WAV는 유지합니다.
5. **짧은 중간 평가:** seed로 고정한 원본 val 200문항(가능하면 Yes/No 절반)을 50 step마다 생성 평가합니다. baseline도 저장된 예측에서 정확히 같은 200문항으로 다시 집계합니다. 모든 trial이 같은 문항을 사용하며, 작은 val의 근소한 차이만으로 최종 결론을 내리지 않습니다.
6. **LR 우선 Optuna:** rank 16에서 LR 5e-5, 1e-4를 먼저 실행하고 세 번째 LR은 TPE가 1e-5~1e-4 로그 범위에서 제안합니다. 각 trial 최대 150 optimizer step, warmup 최대 10 step, 100 step부터 MedianPruner가 이전 trial과 비교해 가지치기합니다. baseline보다 좋은 작은 val 결과가 있으면 그 LR로 rank 32를 추가 비교합니다. 최고 점수가 2회 연속 개선되지 않아도 100 step 이후 종료합니다. 최종 후보 최대 2개를 **전체 val**로 재평가하고 `best.tsv`에 선택합니다.
7. **쉘 + 웹 모니터링:** 모든 자식 stdout/stderr를 쉘/노트북 셀과 `console.log`에 즉시 전달합니다. TensorBoard에서는 loss·LR·gradient norm·검증 점수·step 시간·GPU 사용률·메모리를, Optuna Dashboard에서는 trial 상태·중간 점수·전체 val 결과를 봅니다. 두 서비스 링크는 학습 시작 전에 표시합니다.

| 고정 항목 | 값 |
|---|---|
| Alpha / dropout | 16 / 0.05 |
| Weight decay | 0.1 |
| Effective batch | 64 |
| Precision / attention | BF16 / SDPA |
| 탐색 step | trial당 최대 150 |
| LR trial 수 | 기본 3개, 이후 유망하면 rank 32 하나 |
| 시간 예산 | 준비 포함 6시간, 현재 검증·최종 전체 val 평가는 추가될 수 있음 |

표준 LoRA 배율은 alpha/rank이므로 alpha 16 고정 시 rank 변경에 따른 배율도 달라집니다. 이는 사용자 지정 설정을 유지한 것입니다. throughput 측정은 긴 오디오에서의 보수적인 추정이며, 실제 속도 향상은 실행 환경에서 확인해야 합니다.

```bash
# 새 환경 설치
python -m pip install -r requirements.txt

# 원본 val baseline: 기존 예측 TSV가 일치하면 재사용 가능
python qwen_train.py --mode baseline --data-dir data/processed \
  --output-dir results/baseline_3b

# 모니터링을 별도 쉘에서 실행
python monitoring.py --root results/optuna_budgeted

# 7단계 실행 + Optuna 탐색
python -u tune.py --data-dir data/processed \
  --baseline-metrics results/baseline_3b/val_predictions.metrics.json \
  --output-dir results/optuna_budgeted --trials 3 --max-steps 150 \
  --eval-steps 50 --min-steps 100 --timeout-seconds 21600

# 전체 val로 선택한 체크포인트의 test 평가 (재학습 없음)
python finalize.py --data-dir data/processed \
  --best-tsv results/optuna_budgeted/best.tsv --output-dir results/optuna_final
```

`--skip-rank32`로 추가 rank 실험을 끌 수 있습니다. `--profile-steps`, `--overfit-steps`, `--val-size`, `--max-steps`, `--trials`로 예산을 조절합니다. 데이터나 설정을 바꾸면 새 output 폴더를 사용하세요. 완료된 프로파일·학습 검사는 재사용합니다. 중단된 trial은 실패 상태로 기록하고 동일 파라미터의 새 trial에서 기존 작업 폴더의 마지막 완전한 체크포인트를 재개합니다. 체크포인트가 없으면 원본 모델부터 시작합니다.

탐색 저장소는 **Optuna JournalStorage (`study.journal`)**입니다. SQLite는 쓰지 않으며 `trials.tsv`/`best.tsv`도 함께 기록합니다. 기존 `results/swift_aug_grid/` 결과와 실행은 건드리지 않습니다. `current.json`, 각 작업의 `status.json`, `step_metrics.jsonl`에도 현재 상태가 기록됩니다. 학습 ETA는 남은 optimizer update 기준으로 평가·모델 로딩 시간을 포함하지 않습니다.

## 모니터링 링크

[run_on_colab.ipynb](run_on_colab.ipynb)의 탐색 셀이 TensorBoard와 Optuna Dashboard를 별도 프로세스로 시작하고 브라우저 Colab 출력에 iframe 화면을 표시합니다. localhost 주소는 Colab VM 내부 주소이므로 PC에서 직접 열 수 없습니다. Colab에서는 로그인된 런타임 proxy를 사용하며 새 외부 계정이나 공개 터널이 필요하지 않습니다. VS Code에서 Colab proxy가 지원되지 않으면 같은 런타임을 브라우저 Colab으로 열거나 해당 포트를 전달해야 합니다. 일반 GPU 서버에서는 표시되는 localhost URL을 사용합니다.

**이미 돌아가는 이전 실행을 멈추지 않고 보려면** [monitor_existing.ipynb](monitor_existing.ipynb)를 별도 Colab CPU 런타임에서 열고 실행하세요. 기본 경로는 기존 `results/swift_aug_grid/`입니다. 이 노트북은 Drive의 `logging.jsonl`을 읽기만 하고 새 TensorBoard event로 변환합니다. 학습 커널이 다른 셀을 실행할 수 없어도 사용할 수 있습니다. Drive 동기화 때문에 갱신이 지연될 수 있으며, 첫 로그가 저장되기 전에는 그래프가 비어 있습니다. 이전 grid 실행을 진행 중인 Optuna study로 바꾸지는 않습니다.

쉘에서는 다음처럼 기존 실행을 관찰할 수도 있습니다.

```bash
python monitoring.py --root results/swift_aug_grid --legacy
```

서비스 실행 로그는 `tensorboard_server.log`, `optuna_server.log`, `legacy_bridge_server.log`에 저장됩니다. 재실행하면 같은 런타임에서 해당 경로의 기존 서비스 프로세스를 재사용합니다. 모니터링 프로세스는 GPU 모델을 로드하지 않습니다.

검증: CPU 단위 테스트와 자식 프로세스 로그/가지치기 통신을 검사합니다. GPU 및 ms-swift 실행은 로컬 환경에서 검증할 수 없으며 실제 Colab 실행에서 확인해야 합니다. Optuna가 설치되어 있으면 JournalStorage 재시작 테스트도 실행합니다.

참고: [Optuna Dashboard](https://optuna-dashboard.readthedocs.io/en/stable/getting-started.html), [TensorBoard 노트북 사용](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks), [ms-swift SFT API](https://github.com/modelscope/ms-swift/blob/v3.12.0/swift/llm/train/sft.py).

예측 TSV에서 빈 `prediction`은 Yes/No 이외의 출력으로 오답 처리합니다. `qwen_train.py --mode train`은 이전 직접 학습 루프이므로 새 탐색에는 사용하지 마세요. 원본 데이터 라이선스는 공식 배포 페이지에서 확인하세요.

## 선택 설정으로 약 2 epoch 학습

`bash run_on_server.sh --stage all`은 준비·baseline·탐색 후 선택한 LR/rank로 원본 모델부터 약 2 source epoch 학습합니다. `--stage train`으로 이 단계만 실행할 수 있습니다. 원본 문항당 변형 하나를 선택하는 샘플링을 유지하고 effective batch 64 기준으로 올림한 step 수를 사용합니다. 매 source epoch마다 전체 val을 평가하며, 조기 종료 없이 지정 step을 완료합니다. 학습 결과는 `search/train_2epochs/`에 저장합니다. Trainer 체크포인트는 3,000 optimizer step마다 저장하며, 총 step이 3,000보다 작아도 `final_adapter/`를 별도로 저장합니다. 최종 학습 지표는 같은 search TensorBoard에 표시됩니다. 기존 탐색 어댑터의 test 평가 명령은 `--stage test`입니다.

현재 서버 탐색은 사용자 지정 microbatch 16, gradient accumulation 4 (effective batch 64)로 고정합니다. 프로파일과 trial에서 batch 32/64는 시도하지 않으며 batch 16 OOM 시 중단합니다.

## 중단 복구

- Baseline: 16문항 배치마다 `val_predictions.partial.json`을 원자적으로 저장합니다. 재실행 시 모델 ID와 입력 데이터 지문, 문항 순서·라벨을 확인하고 처리하지 않은 문항부터 예측합니다. 완료 후 기존 TSV와 metrics JSON을 생성합니다.
- 탐색: 50 optimizer step마다 모델·optimizer·scheduler·RNG를 포함하는 Trainer 체크포인트를 저장합니다. 재실행 시 같은 작업 폴더의 마지막 완전한 체크포인트에서 이어갑니다. 중단된 Optuna trial은 FAIL로 남고 같은 파라미터의 새 trial이 해당 작업 폴더를 재사용합니다.
- 최종 학습: 약 1 source epoch 경계와 3,000 optimizer step마다 체크포인트를 저장하고 마지막 `final_adapter`도 저장합니다. 같은 명령을 다시 실행하면 마지막 완전한 체크포인트에서 자동 재개합니다.
- `aqa_state.json`은 체크포인트 저장이 끝난 뒤 기록되며 검증 최고 점수·이력도 복구합니다. 저장 도중 끊긴 체크포인트는 무시합니다. GPU에서 실제 중단·재개 통합 실행은 아직 검증 전입니다.

## 탐색 결과 표 읽기

`results/a100_macro_f1/search/trials.tsv`에는 trial 하나당 한 행이 저장됩니다. 최종 선택한 한 행은 같은 폴더의 `best.tsv`에 저장되며, 이 설정의 LR·rank를 최종 약 2 epoch 학습에 사용합니다. 사용자 지정 output 경로에서는 해당 경로의 `search/`를 확인하세요.

| TSV 항목 | 의미 |
|---|---|
| `number` | Trial 번호 |
| `state` | `COMPLETE`: 정상 완료, `PRUNED`: Optuna 비교에 의해 조기 중단, `FAIL`: 실행 실패 |
| `learning_rate`, `rank` | 해당 trial의 LR과 LoRA rank |
| `microbatch`, `effective_batch_size` | 실제 microbatch와 gradient accumulation을 반영한 유효 batch |
| `value` | COMPLETE는 작은 val 200문항의 최고 macro-F1. PRUNED는 Optuna가 기록한 마지막 중간 점수일 수 있으므로 최고 점수는 평가 이력과 함께 확인 |
| `full_val_value` | 전체 val 1,376문항 macro-F1. 전체 평가를 받지 않은 trial은 빈 값 |
| `best_step` | 작은 val macro-F1이 가장 높았던 optimizer step |
| `stop_reason` | 학습 종료 이유 |
| `adapter` | 최고 작은 val 점수의 어댑터 저장 경로 |
| `intermediate_values` | step별 작은 val macro-F1 이력(JSON) |

작은 val 평가는 기본 50 optimizer step마다 수행합니다. **PRUNED trial은 중단 전까지의 작은 val 평가 결과만 남고, 전체 val 평가는 받지 않으며 최종 선택 후보에서 제외됩니다.** 정상 완료한 `COMPLETE` trial을 작은 val 최고 macro-F1 순으로 정렬해 상위 최대 2개만 전체 val로 재평가합니다. 따라서 `COMPLETE`여도 전체 val 평가 대상이 아니면 `full_val_value`가 비어 있습니다. 최종 선택은 전체 val macro-F1을 기준으로 합니다.

`trials.tsv`는 trial 종료 시 갱신되며, 웹에서는 Optuna Dashboard의 Trials 화면에서 진행 중인 trial의 중간 점수도 확인할 수 있습니다.

## StemFoley 평가

HF 로그인 후 `uv run python -u evaluate_stemfoley.py --batch-size 128`로 실행합니다. 베이스 Qwen, 최종 학습 step 100, 탐색 최고 step 150을 평가하며, 476개 전체와 `in_final_v3` 306개 부분집합의 accuracy·macro-F1을 저장합니다. 세 판정법은 TAD yes/no 토큰 그룹 비교, 같은 그룹 확률의 50% 기준, 선행 공백을 포함한 단일 `yes`/`no` 토큰 비교입니다. 그룹 확률 50% 기준은 TAD 비교와 수학적으로 같은 결과를 냅니다. 예측과 요약은 `results/stemfoley_eval/`에 저장합니다.
