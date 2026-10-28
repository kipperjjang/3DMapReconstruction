# DynaRaySim/torch: world-Z-aligned LiDAR / 200,000 frames

이번 데이터는 `/home/intern/DynaRaySim/torch`, `/home/intern/DynaMapping/torch`,
conda `dyna`로 생성한다. 목표는 **40개 독립 장면 × 5,000프레임 = 200,000개 실제
수집 시각**이며, 새 데이터 전체 상한은 **10,000,000,000 bytes**다.
기존 v1/v2 데이터와 모델은 보존한다. v2는 현재 1,500회 GPU 학습 완료 기록이 있다.

## 수집 상태와 학습 시작

```bash
cd "/home/intern/Neural Scene unet"
conda activate dyna
python scripts/status_dataset_v3.py
```

상태 명령의 frame 수는 로그가 마지막으로 flush된 시점의 보수적인 하한이다.
수집 중에는 shard 이름이 `frames.h5.partial`이며, 정상 종료 후 `frames.h5`가 된다.
완료된 장면만 `sequence.json`을 갖는다. `partial_*_report.json`은 중간 검사이며
최종 완료 표시가 아니다.

전체 생성 후 `outputs/training_dataset_v3/dataset.json`이 생성된다.
`validation_report.json`의 `status=validated`, `total_frames=200000`과
`sample_audit.json`의 `status=passed`로 최종 검증 완료를 확인한다.

사용자가 GPU 호스트에서 본 학습을 실행한다.

```bash
cd "/home/intern/Neural Scene unet"
conda activate dyna
bash scripts/train_v3.sh

# 중단한 동일 설정/데이터의 학습 재개
bash scripts/train_v3.sh --resume
```

`train_v3.sh`는 dyna를 활성화하고 CUDA 사전 검사, 200,000프레임 데이터 검증,
GPU smoke 검사를 통과한 뒤 학습한다. 결과/checkpoint/TensorBoard 이벤트는
`outputs/training_run_v3`에 저장한다. 현재 도구 실행 공간에서는 NVIDIA 장치가
보이지 않으므로 여기서의 본 GPU 학습 시도는 사전 검사에서 중단됐다.

## 데이터 구성

| 항목 | 설정 |
|---|---|
| Train | 32개 장면, 160,000프레임 |
| Validation | 4개 장면, 20,000프레임 |
| Test | 4개 장면, 20,000프레임 |
| 빈도/길이 | 10 Hz, 장면당 500초 |
| 장면 | 랜덤 크기 박스·계단 형태 블록·벽·통로와 단단한 지면 |
| Dynamic objects | 없음. 물체는 정적이며 LiDAR만 이동·회전한다. |
| LiDAR | 명시적 합성 spinning lidar, 32 channels, 수평 360° / 3° 간격, 수직 −85°~45° |
| 자세 | 장착 roll/pitch=0, terrain alignment off, yaw만 변화 |
| GT | 현재 DynaRaySim의 연결 자유공간 경계 표면, boundary 5cm, sample max edge 3cm |
| 학습 grid | robot frame, 64³, 5cm, lower corner (−1.6,−1.6,−1.6)m |

장면마다 geometry, 경로의 끝점·yaw·센서 높이가 다르다. 정적 장면이므로 동일한
경계 GT를 공유하고, 프레임마다 실제 pose로 옮겨 crop/voxel centroid를 계산한다.
모든 관측은 DynaRaySim `runner.run`의 실제 ray casting으로 수집한다. 기존 데이터
복사, 같은 frame 반복, 학습 augmentation을 200,000개에 포함하지 않는다.
장면 내부 pose 중복과 누락/중복 timestamp를 검사하고, 장면별 mesh hash도 검사한다.

센서와 로봇의 기록된 pose로 관측을 robot frame으로 변환한 후 실제 DynaMapping이
이전 frame→현재 frame 변환을 계산한다. 각 frame의 `R_world_sensor[:,2]`가
`[0,0,1]`인지 확인한다. GT voxel 강제 보존, pruning, t=0 병합, detached feedback,
채널 `[32,64,128,256,256]` 등 기존 네트워크 정책은 유지한다.

## 논문의 43%와 이번 커버리지

로컬 `2206.08077v1.pdf` III.D의 43%는 **GT 중 관측되는 비율**이다. 논문은 네 대의
하향 depth camera를 쓰며, 해당 문장의 대응 거리 허용치는 명시하지 않는다.
이번 upright LiDAR와 센서 조건이 다르므로 완전한 재현 수치로 부르지 않는다.

이번에는 모든 frame에서 다음 두 지표를 별도로 기록한다.

1. `gt_point_coverage_5cm`: crop 안의 **dense GT 표면점** 중 현재 관측점까지의
   최근접 거리가 5cm 이내인 점의 비율. 데이터셋 요약은 frame별 비율의 평균이다.
2. `gt_voxel_recall`: GT occupied voxel 중 현재 관측도 같은 voxel을 점유하는 비율.

GT를 관측 가능한 곳으로 줄이지 않는다. 첫 지표는 거리 허용치 때문에 더 높으며
두 지표를 섞어 비교하면 안 된다. 별도 pilot seed의 300프레임에서는 각각
**45.42% / 20.50%**였다. 본 수집의 최종 수치는 `collection_summary.json`과
`validation_report.json`에 기록된다. Pilot은 본 200,000프레임에 포함하지 않는다.

## 저장과 검증

각 장면 폴더에는 authored `scene.obj`, `scenario.yaml`, 공유 `surface_gt.npz`,
압축 `frames.h5`, `sequence.json`, `gt_metadata.json`이 있다. 관측은 crop 안의
raw point 개수와 중복을 유지한다. 이전 prediction과 병합하기 전에 관측을 평균내지 않는다.

HDF5에는 각 점의 3개 uint8 voxel 좌표와 3개 uint8 내부 offset을 gzip으로 저장한다.
내부 offset은 voxel당 256 bins의 중앙값으로 복원하며 3D 위치 오차는 약 **0.17mm 이하**다.
반올림으로 voxel이 바뀌지 않게 저장한다. 네트워크 reader가 float32 metre 좌표로
복원한다. Pose는 float64, timestamp는 int64다. HDF5의 프레임별 offsets로
필요한 구간만 읽고, 학습 캐시는 512프레임으로 제한한다.

전체 검증은 모든 shard의 hash, frame 수, point 개수, GT voxel 유일성, timestamp,
고유 pose, z축 및 DynaMapping 변환을 확인한다. 별도의 대표 프레임 감사는 원본 mesh에서
다시 ray casting하고, 공유 GT 전체를 독립 crop하여 저장된 점과 비교한다.
파일럿 대표 9프레임은 최대 0.167mm 이내에서 재현됐다.

```bash
bash scripts/run_dyna.sh scripts/validate_dataset_v3.py --dataset outputs/training_dataset_v3
bash scripts/run_dyna.sh scripts/audit_dataset_v3.py --dataset outputs/training_dataset_v3
```

## 학습 설정과 환경

본 학습 설정은 20,000 updates, batch 2, rollout 12, position noise ±5mm,
XY mirror, Adam LR 0.01→0.0001, seed 20261028이다. 1,000 updates마다 전체 validation을
평가하고 best를 선택한다. 마지막에 best checkpoint로 test를 한 번 평가한다.
학습에서 480,000 frame을 샘플링하는 것은 train 160,000프레임의 약 세 번 사용에
해당하며, 새로 수집한 200,000 timesteps와는 다른 수량이다.

`dyna`의 Python 3.10 / torch 2.6.0+cu118 / NumPy 2.2.6에 맞춰 MinkowskiEngine을
작업 폴더에 별도로 빌드했다. CPU backend는 `.deps-dyna`, CUDA backend는
`.deps-dyna-cuda`다. CUDA 11.7 toolkit으로 8.6+PTX를 빌드했고 minor version 경고를
로그에 남겼다. CUDA import는 통과했으나 실제 GPU 연산은 호스트에서 검사해야 한다.

네트워크 다운로드가 막혀 SciPy 1.15.3은 호환되는 기존 Python 3.10 설치를 workspace에
연결했다. TensorBoard 2.11.2는 별도 workspace 사본에서 NumPy 2의 문자열 타입 이름에
맞췄다. conda 환경 패키지와 외부 simulator/mapper 소스는 수정하지 않았다.
전체 회귀 41개 및 packed reader/cache 추가 검사 1개가 통과했다. Pilot의 full-model
CPU 학습 업데이트 1회와 validation/test 경로도 완료됐으나 학습 성능 결과로 취급하지 않는다.

수집/빌드/검증 로그와 결정 기록은 `project_log/dataset_v3/`에 보존한다.
