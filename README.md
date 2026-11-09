# Neural Scene sparse U-Net reconstruction

v3 / conda `dyna` 사용 문서:

- [Inference 실행·export·DynaViewer 재생](INFERENCE_GUIDE.md)
- [TensorBoard 접속·지표·저장 파일 설명](TENSORBOARD_GUIDE.md)
- [v3 데이터 구성과 학습 명령](DATASET_V3_GUIDE.md)

`2206.08077v1.pdf`의 Figure 2 및 III.A–C를 참고한 MinkowskiEngine 구현이다.
사용자와 결정한 채널은 `[32,64,128,256,256]`, 단계당 일반 convolution은 2개다.

**입력 정책:** 직전 reconstruction을 DynaMapping 방식으로 현재 robot frame에 정렬하고,
현재 관측은 `t=0`, 정렬된 직전 예측은 `t=1`로 각각 voxelization한다.
같은 공간 voxel의 두 출처도 별도 시간 좌표로 유지한다. 새 모델의 기본 정책은
`aligned_temporal_t01`이며 공간 down/up-sampling에서는 시간 해상도를 유지한다.
최종 `1×1×1×2` convolution(시간 stride 2)이 두 시간을 통합해 `t=0` 점군 하나를 출력한다.
기존 `aligned_merge_t0` 체크포인트는 로딩 시 기존 입력·출력 구조를 자동 선택한다.
기존 v3 데이터는 관측·GT·pose가 분리 저장되어 있어 재수집 없이 사용할 수 있다.

새 시간 구분 모델은 새 출력 폴더에서 학습한다. 기존 가중치를 시간 구분 모델의
resume checkpoint로 사용할 수는 없다. 기존 학습을 resume하면 기존 정책을 유지한다.

```bash
conda activate dyna
bash scripts/train_v3.sh --output outputs/training_run_v3_temporal \
  --input-policy aligned_temporal_t01
```

## 실행

`scripts/run.sh`는 CONDA `sp_conv`의 Python과 현재 폴더의 `.deps`를 사용한다.
`.deps`에는 실제 빌드한 MinkowskiEngine CPU backend가 있다.

```bash
bash scripts/run.sh -m neural_scene smoke
bash scripts/run.sh -m unittest discover -s tests -v
```

`smoke`는 전체 채널 모델로 synthetic stairs의 forward, loss, backward를 확인한다.
가중치는 무작위 초기화이며 reconstruction 성능 평가나 학습된 모델이 아니다.
다른 Python은 `NEURAL_SCENE_PYTHON`으로 지정한다. 환경 변경 시 binary를 다시 빌드한다.

```bash
conda activate sp_conv
python scripts/build_minkowski.py --jobs 2
# CUDA driver와 PyTorch CUDA가 정상인 머신에서:
python scripts/build_minkowski.py --cuda --cuda-home /home/intern/cuda-11.7 --cuda-arch '8.6+PTX' --output .deps-cuda --jobs 2
```

Build는 `/home/intern/MinkowskiEngine` 소스를 `.build`에 복사한 뒤 `.deps`에 배치한다.
CONDA 환경의 패키지와 원본 소스는 수정하지 않는다. CUDA backend 빌드는 완료했고,
실제 GPU 실행 확인은 호스트 터미널에서 진행한다.

## 네트워크

| 단계 | 공간 grid / stride | 채널 | 연산 |
|---|---|---|---|
| 입력 | 64³ / 1 | 3 | 합친 점군의 voxel centroid offset |
| Encoder 0 | 64³ / 1 | 32 | 일반 conv 2개 → spatial downsample |
| Encoder 1 | 32³ / 2 | 64 | 일반 conv 2개 → spatial downsample |
| Encoder 2 | 16³ / 4 | 128 | 일반 conv 2개 → spatial downsample |
| Encoder 3 | 8³ / 8 | 256 | 일반 conv 2개 → spatial downsample |
| Bottleneck | 4³ / 16 | 256 | 일반 conv 2개 |
| Decoder 3 | 8³ / 8 | 256 | generative upsample → skip concatenate → conv 2개 → pruning |
| Decoder 2 | 16³ / 4 | 128 | 동일 |
| Decoder 1 | 32³ / 2 | 64 | 동일 |
| Decoder 0 | 64³ / 1 | 32 | 동일 |
| 출력 | 공간 64³ / 1, 시간 1 slice / 2 | 3 | 1×1×1×2 conv → sigmoid offset → robot-frame 점군 |

일반 kernel `[3,3,3,3]`, down/up kernel `[2,2,2,1]`, stride `[2,2,2,1]`이다.
일반/down/up convolution 뒤에 BatchNorm + ReLU가 붙는다.
Upsampling은 `MinkowskiGenerativeConvolutionTranspose`로 새 voxel을 생성한다.
Skip은 없는 쪽 feature를 0으로 채운 좌표 합집합 concatenate다.

Decoder의 각 별도 1×1 head에서 `sigmoid(logit) >= α`인 voxel만 유지한다. α 기본값은 0.5다.
학습에서도 GT를 강제로 보존하지 않고, 모두 제거되면 빈 점군을 반환한다.
기본 grid는 voxel `0.05 m`, shape `64³`, lower corner `(-1.6,-1.6,-1.6) m`다.
Origin은 논문에 명시되지 않아 `VoxelGrid(origin=...)`로 바꿀 수 있다.
Grid 밖 점은 제거하며 lower boundary는 포함하고 upper boundary는 제외한다.

## API와 학습 loss

아래 API는 위 실행 스크립트를 사용하거나 `PYTHONPATH="$PWD:$PWD/.deps"`를 설정해서 실행한다.

```python
from neural_scene.inputs import merged_input
from neural_scene.model import ReconstructionUNet4D
from neural_scene.loss import reconstruction_loss

model = ReconstructionUNet4D().train()
# 각 변수: batch member별 [N,3] Tensor 목록, 현재 robot frame, metres
result = model(merged_input(current, previous_aligned))
loss = reconstruction_loss(result, ground_truth)
loss.total.backward()
points_by_batch = result.points()
```

Loss는 decoder 단계별 occupancy BCE 평균과, 동일 voxel 좌표의 출력/GT mean Euclidean
offset error다. BCE는 pruning 전 후보를 사용한다. 기본 가중치는 1:1이며 함수 인자로 노출했다.
Offset error는 voxel 단위로, `voxel_size`를 곱하면 metres다. 실제 GT surface 점군을 사용한다.
BatchNorm은 voxel이 2개 이상이면 batch 통계로 학습한다. 희소 support가 하나만 남으면
해당 연산에서만 running 통계를 사용하며 feature/affine gradient는 유지한다.
Pruning 규칙을 바꾸거나 GT voxel을 추가하지 않는다.

## DynaMapping 방식의 순차 처리

```python
from neural_scene.mapping import NeuralSceneMapper, PointCloudFrame

model.eval()
mapper = NeuralSceneMapper(model, frequency_hz=10)
for frame in frames:
    # points: robot frame (m); robot_pose: T_world_robot (4x4 또는 XYZ+XYZW)
    # timestamp_ns: strictly increasing integer ns
    step = mapper.update(frame)
    points_robot = step.points
mapper.reset()  # 새 episode 시작
```

기존 DynaMapping frame 객체도 동일 필드로 받는다. 제공하는 `PointCloudFrame`은 Python 3.7
호환이다. 기존 DynaMapping package의 `typing.Protocol` 의존성은 강제하지 않는다.
정렬은 DynaMapping과 동일한 초기 robot world 기준,
`R_current.T @ R_previous`, `R_current.T @ (p_previous-p_current)`다.
센서 frame 점군은 알려진 extrinsic으로 먼저 robot frame에 옮겨야 한다.
Pose를 추정하거나 보간하지 않는다.

Feedback은 직전 **네트워크 출력**이며 raw scan history가 아니다.
`update(frame, track_grad=True)`도 저장 feedback은 항상 detach한다.
`rollout_loss(mapper, frames, targets)`는 논문의 12-step loss를 계산한다.
`training.adam_with_decay(model, total_updates=...)`는 Adam LR `0.01 → 0.0001`
exponential decay를 제공하며 기간은 직접 전달한다. 직접 API를 쓸 때 optimizer step은 호출자가 수행한다.
데이터 생성과 학습 실행 pipeline은 아래와 같다. 원 논문의 전체 augmentation/data 규모를 재현한 것은 아니다.

## DynaRaySim → DynaMapping → GPU 학습

호스트의 CONDA `sp_conv`에서는 RTX 4090 CUDA가 정상이다. 현재 작업 도구의 실행 공간에는
NVIDIA device nodes가 노출되지 않는다. 승인된 호스트 실행에서는 GPU 접근이 가능하며,
전체 모델 CUDA forward/loss/backward와 실제 500-update 학습·평가를 완료했다.

CUDA backend는 `.deps-cuda`에 별도로 빌드했다. CUDA 11.7과 Ada 호환을 위해
`8.6+PTX`를 사용했다 ([NVIDIA 근거](https://docs.nvidia.com/cuda/ada-compatibility-guide/)).
`run.sh`는 필요한 CUDA 11.7 runtime library 경로도 설정한다.

```bash
# 전체 모델 CUDA forward/loss/backward 확인
NEURAL_SCENE_DEPS="$PWD/.deps-cuda" bash scripts/run.sh -m neural_scene smoke --device cuda:0
# 데이터가 준비되었으면 아래 하나의 명령으로 preflight → smoke → 학습/평가
bash scripts/train_gpu.sh
# 중단한 run 재시작 (50 updates마다 latest checkpoint 저장)
bash scripts/train_gpu.sh --resume
```

기본 학습은 **500 optimizer updates, batch 2, rollout 12**, Adam LR `0.01→0.0001`이다.
관측에 uniform position noise `±0.05 m`를 적용하고 x/y mirror는 각 rollout 전체에
일관되게 적용한다. Mirror 시 상대 pose도 conjugation하여 feedback 정렬을 유지한다.
`--position-noise 0 --no-mirror`로 끌 수 있다. Validation/test에는 augmentation을 적용하지 않는다.
학습 기간은 bounded baseline 실험이며 논문의 200,000-timestep 데이터 학습을 재현하지 않는다.
새 rollout은 관측이 있는 frame에서 시작한다. 이후 관측이 없는 frame은 그대로 포함하며,
직전 reconstruction을 feedback으로 사용한다.

새 archive와 전처리 파일은 `outputs/training_dataset_v1`에 저장한다. 데이터 준비를 독립 실행하려면:

```bash
/home/intern/DynaRaySim/.venv/bin/python scripts/prepare_training_data.py --device cpu
/home/intern/DynaRaySim/.venv/bin/python scripts/validate_training_data.py
```

원본 DynaRaySim/DynaMapping 코드, 기존 archive, CONDA 환경 패키지는 수정하지 않는다.
관측과 composed surface GT archive를 새로 생성하고 각 시나리오의 처음 90 frames를 전처리한다.
Raw 관측을 먼저 voxel 평균하지 않는다. 그래야 prediction과 합칠 때 실제 점 개수 가중치가 유지된다.
GT는 현재 robot frame으로 옮긴 뒤 grid 내부 surface centroid만 저장한다.
Frame 간 transform은 실제 DynaMapping에서 계산해 저장하고 학습의 prediction 정렬에 사용한다.

| Split | 시나리오 |
|---|---|
| Train | gymnasium, indoor partitions, trapezoid, furnished office, furnished lounge |
| Validation | outdoor rough ground |
| Test | outdoor obstacle passage |

시나리오 단위 split이며 같은 궤적의 인접 frame을 다른 split에 섞지 않는다.
Indoor training → outdoor evaluation의 분포 차이를 포함하는 baseline이므로 성능은 report로 판단한다.

학습 결과는 `outputs/training_run_v1`에 저장한다.
GPU preflight와 실행 중 오류를 포함한 새 실행의 터미널 로그는 각 output 폴더의
`console.log`에 저장한다. 기존 v1 로그는 `outputs/gpu_training_console.log`에 보존했다.

- `training.jsonl`: update별 loss, LR, 출력 voxel 수.
- `events.out.tfevents.*`: TensorBoard용 학습 및 평가 지표.
- `initial_validation.json`: 무작위 모델 및 관측만 사용한 baseline.
- `best.pt`, `best_validation.json`: validation F1 기준 학습된 checkpoint.
- `last.pt`, `latest.pt`: 최종 모델 및 optimizer/RNG를 포함한 재시작 checkpoint.
- `report.json`: 완료 후 best checkpoint의 별도 test precision/recall/F1와
  같은 voxel에서의 centroid distance 및 height MAE.

`report.json`의 `status=completed`가 있어야 실제 학습·평가가 끝난 것이다.
작은 fixture에서 학습 경로가 통과한 것과 실제 데이터 학습 완료를 구분한다.

TensorBoard 이벤트는 새 학습부터 모델 checkpoint(`best.pt`, `last.pt`, `latest.pt`)와 같은 `<output>` 폴더에 바로 기록한다.
학습 환경에 의존성이 없으면 프로젝트 내부 경로에 설치한다:

```bash
bash scripts/run.sh -m pip install --target .deps-tensorboard -r requirements-tensorboard.txt
bash scripts/train_v2.sh --output outputs/training_run_v2_tb
# 별도 TensorBoard 환경에서도 같은 로그를 열 수 있다.
bash scripts/run.sh -m tensorboard.main --logdir outputs --port 6006
```

브라우저에서 `http://localhost:6006`을 연다. `train/loss`, `train/occupancy`,
`train/offset`, `train/lr`, `train/output_voxels`, `train/elapsed_s`는 매 update 기록한다.
`validation/model/*`, `validation/measurement_baseline/*`, `validation/per_scene/*`는
초기 step 0과 기존 validation 주기에 기록하며 precision/recall/F1 및 거리 오차를 포함한다.
GT와 일치하는 voxel이 없어 거리 지표가 `null`이면 해당 scalar는 생략한다.
`test/*`는 최종 선택된 best checkpoint의 update에 기록한다. `--evaluate-only`는 step 0을 쓴다.
`--resume`은 같은 이벤트 폴더를 이어 쓰며 checkpoint 이후 미확정 step을 숨긴다.
기존 JSON/JSONL 로그도 유지하며 `--no-tensorboard`로 이벤트 기록을 끌 수 있다.
이미 완료된 run의 과거 JSON 로그를 자동 변환하지는 않는다.

현재 `training_run_v1`은 학습·평가가 완료됐다. Best checkpoint는 update 150이며,
test precision/recall/F1은 0.2075/0.3260/0.2536이다. 관측 baseline F1은 0.2552로,
coverage는 늘었으나 false positives가 많아 F1 개선을 달성하지 못했다.
상세 결과는 [TRAINING_RESULTS.md](TRAINING_RESULTS.md)를 참고한다.

이후 정확도 개선 실험에서는 위치 노이즈를 ±5mm로 줄인 모델이 선택됐다.
동일 test F1은 **0.2536 → 0.3132**, precision은 **0.2075 → 0.3456**,
recall은 **0.3260 → 0.2863**이다. 새 추천 checkpoint는
`outputs/accuracy_study_v2/runs/noise005/best.pt`이며 v1 파일은 보존했다.
실패한 후보를 포함한 비교와 재현 방법은 [ACCURACY_RESULTS.md](ACCURACY_RESULTS.md)에 있다.

## 최신 GT를 사용한 확장 데이터셋 (v2)

현재 DynaRaySim의 PyTorch GT 구현으로 7개 시나리오의 전체 구간을 새로 수집한다.
기존 630프레임 대신 총 1,420프레임(train 1,040 / validation 180 / test 200)을
사용하며, 시나리오별 split은 유지한다. 원시 archive와 학습 NPZ를 합한 새 데이터셋의
상한은 **5,000,000,000 bytes**다. 기존 데이터와 checkpoint는 보존한다.

```bash
/home/intern/DynaRaySim/.venv/bin/python scripts/prepare_dataset_v2.py
/home/intern/DynaRaySim/.venv/bin/python scripts/audit_dataset_v2.py
# GPU가 노출된 환경에서 실행
bash scripts/train_v2.sh
# 중단된 동일 run 재개
bash scripts/train_v2.sh --resume
```

출력은 `outputs/training_dataset_v2`와 `outputs/training_run_v2`다.
수집은 CPU에서도 동작한다. 데이터셋의 `validation_report.json`은 전체 frame 검증,
`sample_audit.json`은 대표 frame의 독립적인 NumPy 좌표 변환/GT 재구성 검사 결과다.
실제 `/home/intern/DynaMapping`으로 정렬하며, 새 API의 sensor-frame 관측을
기록된 extrinsic으로 robot frame에 옮긴 후 `DynaMapping.update`에 전달한다.

저장량을 제한하기 위해 `composed` GT를 사용하므로 동적 접촉면의 근사 오차가 남는다.
과거 auto 해상도 검증 보고서와 최신 ray-pattern fingerprint가 다르면 명시적인
5cm 경계 격자/3cm 최대 삼각형 edge를 사용하고 미검증 상태를 provenance에 기록한다.
대표 frame 검사는 전체 경계 해상도 감사의 대체가 아니다.

학습 설정은 기존 전체 채널 모델, batch 2, rollout 12, 위치 노이즈 ±5mm,
Adam LR 0.01→0.0001, **1,500 updates**, 100회마다 validation이다.
Validation F1으로 best를 선택한 후 test를 평가한다. 새로운 GT와 전체 길이를 쓰므로
이전 90-frame test 결과와 수치를 직접 비교하면 안 된다.
`report.json`의 `status=completed`가 있어야 본 학습·평가 완료다.
GPU가 노출되지 않은 환경에서는 사전 검사에서 중단되며 CPU로 자동 전환하지 않는다.
실행 기록은 [project_log/dataset_v2/progress.md](project_log/dataset_v2/progress.md)에 있다.
현재 생성 결과와 GPU 실행 제한은 [DATASET_V2_RESULTS.md](DATASET_V2_RESULTS.md)에 정리했다.

## DynaRaySim/torch + dyna 확장 수집 (v3)

World z축에 정렬한 LiDAR로 40개 랜덤 장면에서 200,000개 고유 프레임을 수집하는
10GB 제한 파이프라인을 추가했다. 압축 HDF5, GT 기준 관측 커버리지, 실제 DynaMapping
정렬을 사용한다. 상태 확인, 검증 및 사용자가 호스트에서 실행할 학습 명령은
[DATASET_V3_GUIDE.md](DATASET_V3_GUIDE.md)에 있다.

```bash
conda activate dyna
python scripts/status_dataset_v3.py
# 전체 수집/검증 완료 후 GPU 호스트에서:
bash scripts/train_v3.sh
```

## DynaMapping NPZ 추론

`training.save_checkpoint(path, model)`로 저장한 학습된 모델을 사용한다.
Input NPZ는 DynaMapping export의 `current_points`를 읽는다. 직전 출력 NPZ의 `points`를
주면 현재 export의 `transformation`으로 정렬한다. Raw `previous_points`를 대체하지 않는다.

```bash
bash scripts/run.sh -m neural_scene infer \
  --checkpoint trained.pt --input frame20.npz \
  --previous-prediction frame19_reconstruction.npz \
  --output outputs/frame20_reconstruction.npz
```

첫 frame은 `--previous-prediction`을 생략한다. Output NPZ에는 `points`, `confidence`,
`origin`, `voxel_size_m`, `point_frame=current_robot`, checkpoint에 따른 `input_policy`가 저장된다.
기존 출력은 덮어쓰지 않는다. `--origin X Y Z`, `--alpha`, `--device`를 지원한다.
CUDA backend 재빌드 후 `--device cuda:0`을 사용한다.

논문과의 차이 및 검증 결과: [PAPER_NOTES.md](PAPER_NOTES.md).

## 학습 checkpoint evaluation → DynaViewer 재생

`neural_scene.evaluation`은 학습된 checkpoint로 split 전체를 순차 추론하고,
관측/예측/GT와 frame별·전체 지표를 저장한다. 직전 **모델 예측**을 현재 robot frame으로
정렬해 feedback으로 사용하며, scene마다 초기화한다. Validation augmentation은 없다.
기본 checkpoint 예시는 최신 v2 run의 `best.pt`다.

```bash
# CPU backend: GPU가 없는 환경에서도 실행 가능. 출력 폴더는 새 이름 사용.
bash scripts/run.sh -m neural_scene.evaluation export \
  --checkpoint outputs/training_run_v2/best.pt \
  --dataset outputs/training_dataset_v2 \
  --split validation --device cpu \
  --output outputs/my_validation

# CUDA backend로 추론하려면 위 명령에 다음 환경/옵션 사용:
# NEURAL_SCENE_DEPS="$PWD/.deps-cuda" bash scripts/run.sh -m neural_scene.evaluation export \
#   ... --device cuda:0
```

현재 모델 환경은 Python 3.7/PyTorch 1.13, DynaViewer는 Python ≥3.9/PyTorch 2.6이므로
추론과 재생을 분리했다. 재생에는 MinkowskiEngine이나 checkpoint가 필요 없다.
현재 설치된 `/home/intern/DynaRaySim/.venv/bin/python`은 `~/DynaViewer`의 `viewer`를 사용한다.
프로젝트 루트에서 실행한다.

```bash
/home/intern/DynaRaySim/.venv/bin/python -m neural_scene.evaluation play \
  --input outputs/my_validation

# 점 대신 5cm occupied voxel box 표시 (크기는 dataset grid에서 읽음)
/home/intern/DynaRaySim/.venv/bin/python -m neural_scene.evaluation play \
  --input outputs/my_validation --voxels

# 예측과 관측을 비교
/home/intern/DynaRaySim/.venv/bin/python -m neural_scene.evaluation play \
  --input outputs/my_validation --reference observation
```

기본 색은 **청록=예측, 녹색=관측, 주황=GT**다. `--foreground observation --reference target`은
관측/GT 비교, `--reference none`은 단독 표시다. DynaViewer의 current/history 레이어를
같은 시각의 비교 레이어로 사용하며, 주황은 과거 frame을 의미하지 않는다.
모든 점은 현재 robot 좌표계(m)이고 카메라는 robot 원점에 고정된다.
VisPy는 마우스로 회전·확대, `R`로 중심 복원, `Esc`로 종료한다.
`--fps 10`, `--loop`, `--no-hold`, `--scene SCENE`를 지원하며 기본은 마지막 frame에서 유지한다.
`--max-points`는 표시량만 제한하고 저장/평가 결과에는 영향을 주지 않는다.

점군별 불투명도와 GT 원본 mesh 표시:

```bash
conda activate dyna
python -m neural_scene.evaluation play --input outputs/my_validation \
  --layers prediction observation target \
  --prediction-opacity 0.9 --observation-opacity 0.7 --gt-opacity 0.15 \
  --mesh --mesh-opacity 0.25
```

VisPy 창에서 `1`=예측, `2`=관측, `3`=GT, `4`=mesh를 선택한 뒤
`+`/`-`로 불투명도를 10%씩 바꾼다. `0`은 선택 레이어를 숨기거나 복원한다.
왼쪽 위에 현재 값이 표시되며 마지막 frame 유지 중에도 조절할 수 있다.
0은 숨김, 1은 불투명이고 voxel의 면과 테두리에도 적용된다.
Rerun은 실행 시 지정한 불투명도를 기록하며 레이어별 entity로 표시한다.

새 v3 export에는 GT 수집에 사용한 `scene.obj`의 geometry를 함께 저장한다.
기존 export는 `--mesh auto --dataset /path/to/training_dataset_v3`으로 원본을 찾거나,
`--scene scene_0030 --mesh /path/to/scene.obj`로 직접 지정한다.
직접 지정하는 mesh는 **world 좌표, metre 단위**여야 하며 재생 시 현재 robot 좌표로 변환된다.
OBJ와 내장 NPZ는 추가 의존성 없이 읽고, 다른 형식은 `trimesh` 설치가 필요하다.
v2처럼 원본 장면이 여러 object로 구성된 경우 world 좌표로 합성한 mesh를 직접 지정한다.

출력 폴더:

- `evaluation.json`: checkpoint/dataset SHA256, grid, feedback 설정, frame별 지표와 파일 목록.
- `report.json`: 기존 학습 evaluator와 같은 voxel precision/recall/F1, matched centroid distance,
  height MAE, 관측 baseline, scene별 지표. 거리는 동일 occupied voxel에서만 계산한다.
- `scene_000/frame_00000.npz` 등: `prediction_points`, `confidence`, `current_points`,
  `target_points`, `robot_pose`, `transformation`, `timestamp_ns`, `frame_id`.

`--limit 3`은 각 scene의 처음 3 frame만 평가한다. `--scene`은 해당 split 안에서 선택하며,
`--split test`로 test 평가도 가능하다. `--no-feedback`은 관측만 입력하는 ablation이다.
기존 출력은 덮어쓰지 않으며, 중단된 export는 `status=running`으로 남아 재생에서 거부한다.
CPU/CUDA 연산 및 pruning 경계 차이로 기존 CUDA 학습 보고서와 수치가 달라질 수 있으므로,
동일 backend에서 전체 구간으로 비교한다.

별도 viewer 환경을 만들 때는 그 환경에서 로컬 패키지를 설치한다:

```bash
python -m pip install -e "$HOME/DynaViewer[vispy]"
# 선택 사항: Rerun 재생/파일 기록
python -m pip install -e "$HOME/DynaViewer[rerun]"
python -m neural_scene.evaluation play --input outputs/my_validation \
  --renderer rerun --headless --recording outputs/my_validation.rrd
```

Rerun은 선택 dependency이며 현재 DynaRaySim 환경에는 설치되어 있지 않다.
