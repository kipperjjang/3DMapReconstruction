# 진행 기록

## GPU 진단

- Kernel: 5.15.0-139-generic; NVIDIA modules loaded, version 535.230.02.
- `/proc/driver/nvidia/gpus/0000:3b:00.0/information`: RTX 4090, excluded=False.
- CUDA toolkit 11.7, sp_conv Torch 1.13.1+cu117.
- 현재 실행 공간에는 `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`이 없음.
- Mount info: host `/dev` 위에 현재 공간의 tmpfs `/dev` mount가 덮여 있음.
- `nvidia-smi` failure, Torch CUDA initialization: No CUDA GPUs are available.
- 따라서 현재 증거는 GPU 장치 미노출을 가리킨다. Host GPU runtime 정상 여부는
  사용자에게 일반 터미널의 nvidia-smi / Torch CUDA 결과를 요청해 대기 중이다.
- `strace`는 이 실행 공간에서 ptrace가 허용되지 않아 사용할 수 없었다.

## 빌드 준비

- CUDA source와 dependency output을 CPU build와 분리하는 옵션을 추가했다.
- CUDA 11.7은 Ada-native sm_89를 생성하지 않으므로 `8.6+PTX`로 빌드할 예정이다.
  NVIDIA 공식 근거: https://docs.nvidia.com/cuda/ada-compatibility-guide/
- GPU 접근이나 driver를 우회/수정하지 않는다. Data 생성/실제 학습은 GPU 확인 이후다.

## Host 확인과 CUDA build

- 사용자가 일반 host 터미널에서 Torch 1.13.1+cu117, CUDA=True, count=1,
  RTX 4090을 확인했다. Host runtime은 정상이고 현재 명령 도구의 device exposure만 문제다.
- CUDA 11.7 / `8.6+PTX` backend build 성공, `.deps-cuda`에 분리 저장했다.
- 새로운 extension의 libcusparse 검색을 위해 run.sh에 toolkit lib64 경로를 추가했다.
- CUDA extension import: ME 0.5.4, CUDA backend=True. GPU runtime 실행은 현재 공간에서 불가.
- CUDA backend의 CPU 경로로 기존 24 tests 통과. Host 전체 모델 GPU smoke 결과는 사용자 확인 대기.

## Data 및 training pipeline

- 새 DynaRaySim composed-GT archive를 workspace에서 CPU로 생성 중이다.
- DynaMapping 실제 reader/update로 preprocessing, 처음 90 frames/scene 저장.
- Train 5 scenes, validation/test 각 1 scene. 전체 시나리오 단위 split.
- 7개 전체 시나리오 archive 생성 및 실제 DynaMapping 전처리 완료.
  Train 450 / validation 90 / test 90, 총 630 frames, workspace 사용량 701 MB.
- 실제 frame의 저장 상대 pose와 DynaMapping-compatible 계산을 직접 비교해 일치 확인.
- 기본 training: 500 updates, batch 2, rollout 12; 모든 입력 t=0, prediction detach,
  LR .01→.0001, position noise ±.05m 및 trajectory-wide XY mirror.
- 각 timestep에서 loss/12 backward하여 12개 graph를 동시에 보존하지 않는다.
- Baseline/validation checkpoint selection/test evaluation/resume pipeline 작성.
- 작은 fixture의 실제 optimizer update, model/optimizer checkpoint, 별도 validation/test report 경로 통과.
- 실제 dataset의 GPU training은 아직 시작하지 않았다.
- 관측이 없는 frame에서 빈 recurrent state로 시작하면 sparse 후보가 없어 학습 신호도 없다.
  관측이 있는 시작점만 선택하고 이후 blind frame은 유지한다. 이 동작 포함 experiment 4 tests 통과.
- 전체 prepared frame의 grid/GT uniqueness/timestamps/relative pose를 검사하는 validator 추가.
- 전체 630 frames 최종 검증 통과. 모든 frame의 feedback transform 최대 오차 0.0,
  중복 GT voxel/영역 밖 좌표/비유한 좌표 없음. Timestamp 단조 증가 확인.
- 결과: outputs/training_dataset_v1/validation_report.json.
- Host 실행 요청 명령: bash '/home/intern/Neural Scene unet/scripts/train_gpu.sh'.
  Preflight / dataset validation / full-model GPU smoke / train / held-out test 순으로 실행.
  전체 콘솔 로그는 outputs/gpu_training_console.log에 저장한다.

## 승인된 host GPU 실행 및 실제 학습

- 사용자가 CLI 권한을 변경하고 GPU 확인 후 학습 계속을 요청했다.
- Sandbox 내부 GPU check는 여전히 device nodes 없음으로 실패했다.
- 승인된 host 실행에서 RTX 4090, Torch CUDA=True 및 GPU matmul backward 확인.
- 전체 모델 GPU smoke 통과: 47,537,383 parameters, loss 1.1543080806732178,
  forward/loss/backward 성공, output 1728 points.
- 실제 training_run_v1 시작. Initial validation model F1=0, measurement F1=0.1886.
- 50 updates: validation F1 약 0.0020, best/latest checkpoint 생성.
- 초반 predictive pruning이 전체 출력을 제거하기도 했으나 이후 출력/offset 학습 재개.
  최종 성능은 500 updates 종료 및 held-out test report 후 판단한다.

## 학습 완료

- Update 456에서 downsample support 한 row로 BatchNorm training variance 오류 발생.
- SparseBatchNorm은 singleton 호출만 누적 running 통계로 정규화하며 feature/affine
  gradient를 유지한다. 일반 batch 동작/채널/학습 및 추론 pruning α=0.5는 유지.
- Resume 시 atomic checkpoint 이후 uncommitted log rows를 제거하도록 수정.
- 새 경계 사례 포함 전체 30 tests 통과, host에서 update 450 checkpoint 재개.
- 500 updates 완료, report.json status=completed, best validation checkpoint update 150.
- Test precision 0.2074922, recall 0.3259943, F1 0.2535819.
- Measurement baseline precision 0.9486813, recall 0.1474163, F1 0.2551800.
- Coverage 증가에도 false positives가 많아 F1 baseline 개선 미달성.
- Matched voxel centroid distance 0.0212666m, matched height MAE 0.0102827m;
  이 offset metric은 matched subset의 오차로 전체 누락/오검출을 포함하지 않는다.
- 최종 무결성 확인: update 1..500 정확히 한 번씩 기록, 모든 loss finite, LR endpoint .0001,
  best/last/latest checkpoint와 completed report 저장. Best 모델 첫 두 test frames 추론
  1560/1684 points NPZ 출력, 두 번째는 첫 번째 prediction feedback으로 정렬/병합.
