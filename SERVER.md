# A100 80GB 서버 실행

이 폴더 전체를 서버로 복사합니다. Python 3.12, NVIDIA 드라이버와 CUDA GPU를 사용할 수 있는 PyTorch 2.7 이상 3 미만, `curl`, `ffmpeg`가 필요합니다. 데이터·모델 캐시·증강 WAV·어댑터는 서버의 로컬 SSD에 두세요.

```bash
cd /path/to/training
# uv 설치가 필요하면: https://docs.astral.sh/uv/getting-started/installation/
bash setup_server.sh
```

`setup_server.sh`는 `uv venv --python 3.12 .venv`로 환경을 만들고 `uv pip --torch-backend auto`로 서버에 맞는 PyTorch backend와 의존성을 설치합니다. 기존 `.venv`는 재사용합니다. 실행 스크립트가 `.venv/bin/python`을 자동으로 사용하므로 activate는 필요 없습니다. CUDA wheel index를 지정하려면 `AQA_TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_server.sh`처럼 실행하세요. 해당 CUDA 버전은 서버 드라이버가 지원해야 합니다. 기존 환경에서 `torchao` 호환성 오류가 발생하면 `uv pip uninstall --python .venv/bin/python torchao` 후 다시 실행하세요. 기본 학습은 양자화를 사용하지 않습니다.

```bash
# 공식 데이터 다운로드/검증/증강 → baseline → 탐색 → 전체 val 비교 → test
CUDA_VISIBLE_DEVICES=0 bash run_on_server.sh \
  --data-root /local_ssd/clotho_aqa/data \
  --output-root /local_ssd/clotho_aqa/results/a100_macro_f1
```

기본값은 LR trial 3개, trial당 최대 150 optimizer step, 50 step마다 val 200문항 평가, 100 step부터 pruning, 탐색 준비 포함 6시간 예산입니다. 데이터 준비·baseline·최종 전체 val·test 시간은 별도입니다. `--timeout-seconds 0`으로 탐색 시간 제한을 해제할 수 있습니다. 탐색에서 선택한 어댑터로 test를 평가하며 전체 재학습 단계는 없습니다.

단계별 실행도 지원합니다. 다음 명령들은 기본 경로인 `data/`, `results/a100_macro_f1/`를 사용합니다. 사용자 경로를 썼다면 매번 동일한 `--data-root`, `--output-root`를 붙이세요.

```bash
bash run_on_server.sh --stage prepare
bash run_on_server.sh --stage baseline
bash run_on_server.sh --stage search
bash run_on_server.sh --stage test
```

SSH 연결 종료 후에도 실행하려면 다음처럼 실행합니다.

```bash
nohup bash run_on_server.sh > server.log 2>&1 &
tail -f server.log
```

서버용 연산 설정:

- Optimizer: 일반 **AdamW (`adamw_torch`)**, 기존 LR·weight decay 유지.
- BF16, SDPA attention, FP32 연산의 TF32 허용. SDPA backend는 입력 조건에 따라 PyTorch가 선택합니다.
- 실제 batch는 64부터 OOM 시 감소하며 gradient accumulation으로 effective batch 64 유지.
- 긴 오디오로 batch를 검증하고 workers 0/2, checkpointing 켜기/끄기를 실측해 성공한 설정 중 가장 빠른 것을 선택.
- pinned memory, worker가 있으면 persistent workers와 prefetch 사용. CPU 연산 스레드는 기본 4개로 제한하며 `AQA_CPU_THREADS=8`처럼 변경 가능.
- 정밀 GPU 동기화는 속도 프로파일에서만 사용. 일반 학습 step 시간과 ETA는 근사치입니다.
- 평가 오디오 waveform은 프로세스 내 최대 128개 캐시. 평가 점수는 accuracy·macro-F1, 선택·pruning은 macro-F1.

모든 trial은 동일한 seed로 원본 순서를 섞고 각 문항의 7개 변형 중 하나를 선택한 입력 스트림을 사용합니다. 데이터 순서를 임의로 길이순 재배열하지 않습니다.

TensorBoard와 Optuna Dashboard는 검색 단계 시작 시 자동으로 켜집니다. 출력된 포트를 SSH로 전달하세요. 기본 포트가 사용 중이면 다른 포트가 출력됩니다.

```bash
# 개인 PC의 별도 터미널에서 실행
ssh -N -L 6006:127.0.0.1:6006 -L 8080:127.0.0.1:8080 user@server
```

`http://localhost:6006`, `http://localhost:8080`에서 확인합니다. 모니터링이 필요 없으면 `--no-monitor`를 붙이세요.

중단 후 같은 명령으로 재실행하면 완료된 작업을 재사용하고 중단된 trial은 마지막 완전한 50-step 체크포인트에서 optimizer·scheduler·RNG와 함께 재개합니다. 최종 학습도 저장된 체크포인트에서 자동 재개합니다. 체크포인트가 없으면 원본 모델부터 시작합니다. 이전 balanced accuracy 탐색 결과와 혼합하지 말고 새 output 폴더를 사용하세요. 설정을 바꾼 실험도 새 폴더가 필요합니다. 서버 경로가 달라졌다면 `--stage prepare`로 오디오 절대 경로를 다시 생성하세요.

실제 A100 throughput과 GPU 학습은 이 코드 작성 환경에서 검증하지 못했습니다. `search/performance.json`에 해당 서버의 실측 결과가 저장됩니다.

연산 설정 참고: [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention), [ms-swift 3.12.0 학습 인자](https://github.com/modelscope/ms-swift/blob/v3.12.0/swift/llm/argument/train_args.py).
