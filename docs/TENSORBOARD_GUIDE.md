# v3 TensorBoard 확인과 저장 내용

이 문서는 `neural_scene.experiment`와 `training.py`의 현재 기록 방식 및
`../outputs/training_run_v3`, `outputs/training_run_v3_cp`의 실제 event tag를 확인해 작성했다.
현재 event에는 scalar가 기록되며 점군·이미지·가중치 histogram·모델 graph는 기록되지 않는다.

## 1. TensorBoard 실행

프로젝트 루트에서 실행한다. 실제 확인할 학습 폴더를 `RUN_DIR`에 지정한다.

```bash
cd "/home/intern/Neural Scene unet"
conda activate dyna
RUN_DIR="outputs/training_run_v3"
# 복사한 학습 기록을 확인한다면:
# RUN_DIR="outputs/training_run_v3_cp"

bash scripts/run_dyna.sh -m tensorboard.main \
  --logdir "$RUN_DIR" \
  --host 127.0.0.1 \
  --port 6006
```

브라우저에서 **http://localhost:6006** 에 접속한다.
이 터미널은 TensorBoard 서버를 실행한 상태로 두고, 종료할 때 `Ctrl+C`를 누른다.
TensorBoard를 종료해도 학습은 계속된다.
`run_dyna.sh`는 workspace의 TensorBoard 및 NumPy 호환 의존성 경로를 설정한다.

두 폴더를 같은 화면에서 비교하려면 다음처럼 실행한다.

```bash
bash scripts/run_dyna.sh -m tensorboard.main \
  --logdir_spec v3:outputs/training_run_v3,v3_copy:outputs/training_run_v3_cp \
  --host 127.0.0.1 --port 6006
```

복사한 폴더는 원본과 별도 파일이므로 이후 학습 기록이 자동 동기화되지는 않는다.
동일 시점의 사본이라면 두 곡선이 겹칠 수 있다.

## 2. 원격 서버에서 접속

서버에서 위 TensorBoard 명령을 실행한다. 로컬 PC의 별도 터미널에서 SSH 터널을 연다.
아래 호스트 이름은 로컬 PC가 접속할 수 있는 서버 주소 또는 SSH alias로 바꿀 수 있다.

```bash
ssh -N -L 6006:127.0.0.1:6006 intern@drcdserver2-PowerEdge-T640
```

로컬 브라우저에서 **http://localhost:6006** 에 접속한다.
로컬 6006 포트가 사용 중이면 `-L 16006:127.0.0.1:6006`을 사용하고
브라우저 주소를 `http://localhost:16006`으로 바꾼다.

## 3. 먼저 확인할 그래프

Scalars 화면에서 아래 tag를 검색한다. 가로축 `Step`은 optimizer update 번호다.
센서 frame 수와 다르며, 기본 v3 설정은 한 update에 batch 2 × rollout 12를 처리한다.

| Tag | 의미와 읽는 방법 |
|---|---|
| `train/loss` | 학습 total loss. 기본값은 occupancy loss + offset loss |
| `train/occupancy` | decoder의 비어 있지 않은 각 단계에서 계산한 occupancy BCE의 평균 |
| `train/offset` | 같은 voxel에서 매칭된 예측·GT의 centroid offset Euclidean error 평균. **voxel 단위**이며 m 환산은 ×0.05 |
| `train/output_voxels` | rollout 동안의 최종 출력 voxel 개수 평균. 각 timestep에서 batch 전체를 합친 개수 |
| `train/lr` | optimizer update 후 scheduler가 갱신한 learning rate |
| `train/elapsed_s` | 이번 실행에서 학습 loop 진입 후 경과 초. 재개 시 다시 시작하며 초기 validation 시간은 제외 |
| `validation/model/f1` | 모델의 validation voxel F1. `best.pt`를 선택하는 기준 |
| `validation/model/precision` | 예측 occupied voxel 중 GT도 점유하는 비율 |
| `validation/model/recall` | GT occupied voxel 중 예측이 복원한 비율 |
| `validation/measurement_baseline/f1` | 현재 관측만 voxel화한 baseline F1. 모델의 복원 효과를 비교하는 기준 |

학습 loss는 각 timestep에서 batch를 평가하고 rollout 12회에 걸쳐 평균한 값이다.
`train/offset`에서 매칭이 없으면 offset 항은 0이 될 수 있으므로,
offset이 낮다는 이유만으로 복원이 좋다고 판단하면 안 된다. F1과 출력 개수도 함께 본다.

TensorBoard의 smoothing을 낮추거나 0으로 설정하면 기록된 변동을 직접 볼 수 있다.
학습 loss와 validation F1은 서로 다른 데이터·지표이므로 같은 척도로 비교하지 않는다.

## 4. Validation·test의 나머지 scalar

| Tag suffix | 의미 |
|---|---|
| `tp`, `fp`, `fn` | 같은 voxel 점유 기준 true positive, false positive, false negative 합계 |
| `frames` | 평가한 frame 수 |
| `matched_distance_sum_m` | 매칭된 voxel들의 centroid 3D 거리 합(m) |
| `matched_height_sum_m` | 매칭된 voxel들의 centroid Z 오차 절댓값 합(m) |
| `matched_centroid_distance_m` | 위 3D 거리 합 / TP. 매칭된 voxel에서만 계산 |
| `matched_height_mae_m` | 위 Z 오차 합 / TP. 매칭된 voxel에서만 계산 |

전체 precision·recall·F1은 전체 frame의 TP·FP·FN을 합쳐 계산한다.
프레임별 F1의 단순 평균은 아니다. Centroid 오차는 전체 점군 Chamfer distance가 아니며,
복원하지 못한 voxel의 영향은 recall/F1에서 확인한다.

같은 항목들이 다음 prefix 아래에 기록된다.

| Prefix | 대상 |
|---|---|
| `validation/model/` | validation 전체의 모델 예측 |
| `validation/measurement_baseline/` | validation 전체의 현재 관측 baseline |
| `validation/per_scene/scene_0000/` 등 | 해당 validation scene의 모델 예측 |
| `test/model/`, `test/measurement_baseline/`, `test/per_scene/.../` | 학습 종료 후 best 모델로 수행한 test. `--skip-test`이면 기록하지 않음 |

TP가 0이면 centroid 평균 오차는 JSON에서 `null`이며, 해당 시점에는 TensorBoard
scalar로 추가되지 않는다. 현재 logger는 유한한 숫자만 기록한다.
이미 존재하는 centroid tag가 특정 step에서 빠져 있어도 반드시 기록 오류는 아니다.

데이터 수집 때 계산한 `gt_point_coverage_5cm`와 strict voxel recall은 별도 지표다.
이 수집 통계는 `../outputs/training_dataset_v3/collection_summary.json`에 있으며
현재 학습 event에 직접 기록되지 않는다.

## 5. 기록 주기와 재개

- `train/*`: 매 optimizer update. 콘솔 출력은 첫 update 및 10 update마다이므로 주기가 다르다.
- `validation/*`: 초기 모델은 step 0, 이후 기본 v3 설정에서 1,000 update마다 및 마지막 update.
- `test/*`: 학습 종료 후 선택된 best 모델의 step 번호에 기록한다. 마지막 update 번호와 다를 수 있다.
- event writer는 약 10초 간격으로 flush하며 validation 후에도 명시적으로 flush한다.
  브라우저 갱신·평가 소요 시간 때문에 최신 step이 바로 보이지 않을 수 있다.
- `--resume`은 checkpoint 다음 step 이후의 미확정 event를 TensorBoard에서 숨기도록
  `purge_step`을 설정한다. 새 event 파일이 추가되는 것은 정상이다.

초기 validation이나 주기적 validation 진행 중에는 train step이 한동안 증가하지 않는다.
v3 전체 validation은 20,000프레임이며, 완료 후 지표를 한 번에 기록한다.
작은 event 파일 여러 개는 초기 실패·재시작으로 생성된 파일일 수 있다.
한 run 폴더를 통째로 지정하면 TensorBoard가 파일들을 함께 읽는다.

## 6. 학습 폴더에 저장되는 파일

| 파일 | 저장 내용과 용도 |
|---|---|
| `events.out.tfevents.*` | TensorBoard scalar 시계열. checkpoint 가중치는 포함하지 않음 |
| `training.jsonl` | update마다 한 줄의 loss, occupancy, offset, output_voxels, step, lr, elapsed_s |
| `console.log` | `train_v3.sh`의 사전 검사·smoke·학습 stdout/stderr, 실행 시 append |
| `configuration.json` | 데이터 manifest SHA256, update·batch·rollout·seed·LR·noise·채널·pruning 등 실행 설정 |
| `initial_validation.json` | 학습 전 전체 validation 결과 |
| `validation_01000.json` 등 | 해당 update의 전체 validation, 관측 baseline, scene별 지표 |
| `best_validation.json` | 최고 validation F1을 얻은 update 번호와 지표 |
| `best.pt` | 최고 validation F1의 모델 가중치, BatchNorm 상태, 모델 config, 포맷·입력 정책 |
| `latest.pt` | validation 후 저장한 학습 재개 상태: 모델·optimizer·scheduler·step·NumPy/Torch/CUDA RNG·metadata |
| `last.pt` | 전체 학습 loop 완료 시 마지막 모델. best와 다를 수 있음 |
| `report.json` | 학습 완료 후 설정·best validation·initial validation·최종 test 등의 요약 |
| `source/`, `source_hashes.json` | 해당 출력 폴더에서 처음 저장한 코드 snapshot과 파일 SHA256. 기존 snapshot은 재개 시 교체하지 않음 |

`best.pt`, `last.pt`는 inference용이며 optimizer가 없다. `latest.pt`가 resume용이다.
`latest.pt`는 매 update마다 저장되지 않으므로 재개 시 마지막 validation 이후 update를
다시 수행할 수 있다. checkpoint는 임시 파일 작성 후 교체한다.
`report.json`·`last.pt`가 아직 없더라도 학습 진행 중이면 정상이다.

## 7. Event 확인·보관·문제 해결

화면 없이 event tag와 기록 상태를 확인하려면 다음 명령을 사용한다.

```bash
bash scripts/run_dyna.sh -m tensorboard.main \
  --inspect --logdir outputs/training_run_v3
```

event에는 scalar만 있으므로 점군을 보려면 별도의 inference export와 viewer를 사용한다.
사용 방법은 [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)에 정리했다.

| 증상 | 확인할 내용 |
|---|---|
| 화면이 비어 있음 | `--logdir`의 run 폴더와 `events.out.tfevents.*` 존재 여부, 초기 validation 완료 여부 |
| train 그래프 갱신이 멈춤 | 콘솔이 validation 중인지 확인. writer flush와 브라우저 갱신 시간을 기다림 |
| validation에 점이 적음 | 기본 1,000 update 간격. train과 기록 주기가 다름 |
| import/NumPy 타입 오류 | `python -m tensorboard` 대신 위의 `bash scripts/run_dyna.sh -m tensorboard.main` 사용 |
| 포트 사용 중 | 서버 포트를 6007 등으로 변경하고 SSH tunnel 목적지 포트도 맞춤 |

곡선만 보관하려면 run 폴더의 모든 `events.out.tfevents.*`를 함께 보관한다.
정확한 수치 확인에는 `training.jsonl`과 validation JSON을 함께 보관하는 편이 좋다.
다시 inference하거나 학습을 재개하려면 목적에 맞는 checkpoint도 필요하다.

관련 코드: [training.py](../neural_scene/training.py), [experiment.py](../neural_scene/experiment.py).
