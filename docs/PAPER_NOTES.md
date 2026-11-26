# 논문 사양, 사용자 결정 및 검증

출처: `2206.08077v1.pdf`, Hoeller et al., *Neural Scene Representation for
Locomotion on Structured Terrain*, Figure 2 및 III.A–C (pp. 3–4).

| 항목 | 논문 | 최종 구현 |
|---|---|---|
| 공간 영역 | 3.2³ m, 64³ cells | 동일, origin은 configurable |
| Voxel 크기 | 0.05 m | 동일 |
| 입력 feature | centroid offset 3채널 | 동일 |
| 이전 출력 정렬 | 현재 robot frame으로 transform | DynaMapping 계산식 참조 |
| 시간 좌표 | 현재 0, 직전 reconstruction 1 | **현재 0, 정렬된 직전 reconstruction 1을 유지** |
| Backbone | 4D fully convolutional U-Net | 두 시간 slice에서 공간·시간 convolution 수행 |
| Encoder | 공간 stride 2 × 4, 시간 stride 1 | 동일 |
| Decoder | generative upsampling + pruning | 동일 |
| Occupancy | 별도 conv + sigmoid | 1×1×1×1 head |
| Pruning | 확률이 α 미만이면 제거, α=0.5 | 동일; GT 보존 및 fallback 없음 |
| 출력 feature | offset 3채널 | 1×1×1×2 conv, 시간 stride 2 + sigmoid; 출력은 t=0 한 slice |
| Loss | 각 scale BCE + Euclidean offset error | 동일 항목, BCE stage 평균, 가중치 configurable |
| Rollout | 12 timesteps, 시간 gradient 전파 없음 | helper 제공, 직전 출력 detach |
| Optimizer | Adam, exponential LR 0.01 → 0.0001 | helper 제공, 기간은 필수 인자 |

## 사용자와 확정한 문서 미기재 아키텍처

- 채널 `[32,64,128,256,256]`.
- Encoder/bottleneck/decoder block의 일반 convolution **2개**.
- 일반 kernel 공간 3·시간 3; down/up kernel 공간 2·시간 1.
- 일반/down/up convolution 뒤 BatchNorm + ReLU.
- Skip은 좌표 합집합의 zero-fill feature concatenate.
- 최종 offset activation은 sigmoid.
- 이전 예측을 현재 robot frame으로 정렬한 후 두 출처를 각각 voxelization한다.
  현재 관측 t=0, 이전 예측 t=1을 encoder/decoder에서 유지한다.
- 최종 offset head의 시간 kernel 2·stride 2로 현재 시점의 한 점군으로 통합한다.
  이 시간축 통합 kernel/stride는 이번 구현에서 명시적으로 선택한 설정이다.
- Decoder occupancy BCE는 두 시간의 후보 모두에 현재 scene의 공간 GT를 적용한다.
  최종 출력 confidence는 같은 공간 voxel의 두 시간 점유 확률 중 최댓값이다.

Grid origin, loss weights, training 기간, batch size는 API로 노출했다. 기본 origin은
robot 중심 lower corner, loss weights는 1:1이다. 원 논문에서 복원한 값으로 주장하지 않는다.
원 논문 전체의 동일 재현으로 주장하지 않는다. 이전 t=0 병합 모델과 학습 결과는
baseline commit `7dee8a4` 및 기존 체크포인트로 보존한다. 아래 검증 수치는
각 기록 시점의 기존 모델 결과이며 새 시간 구분 모델의 학습 성능이 아니다.

## DynaMapping 참조

`/home/intern/DynaRaySim/DynaMapping/dynamapping/core.py`의 `update()`와
`geometry.py`를 확인했다. Pose는 `T_world_robot`, quaternion은 XYZW,
점군은 robot frame, 단위는 m다. Initial robot pose 기준으로 두 pose를 표현한 뒤
`R_current.T @ R_previous`, `R_current.T @ (p_previous - p_current)`로 변환한다.

`neural_scene.geometry.previous_to_current()`는 같은 계산을 사용한다.
직전 raw scan 대신 직전 reconstruction을 feedback으로 사용한다. DynaMapping 원본은 수정하지 않았다.

## 검증

- CONDA `/home/intern/anaconda3/envs/sp_conv`.
- Python 3.7.16, PyTorch 1.13.1+cu117, NumPy 1.21.6.
- `/home/intern/MinkowskiEngine` 소스로 workspace `.deps`에 CPU backend 빌드 성공.
  원본과 CONDA 환경 패키지는 보존했다.
- 호스트 터미널에서 Torch CUDA=True, GPU 1개, RTX 4090 확인.
  현재 명령 도구 공간에는 NVIDIA device nodes가 없어 GPU 실행 불가.
- CUDA 11.7 / `8.6+PTX` backend를 `.deps-cuda`에 별도로 빌드하고 import 성공.
- 전체 모델 **47,537,383 parameters**.
- Full-channel smoke: 134 input voxels, decoder 후보 160/419/1687/2543,
  retained 40/202/304/1539, loss 1.1188173, forward/loss/backward 성공.
- Unit/integration 30개 통과: centroid/crop/pose/merged t=0/detached feedback,
  batch identity/union skip gradients/generative support/four decoder scales,
  occupancy 및 offset gradients/empty input/complete pruning,
  checkpoint round-trip/LR endpoint/NPZ CLI 정렬 및 추론.
- 추가 experiment 검증: 작은 fixture의 optimizer update/checkpoint/validation/test,
  scene split 중복 거부, mirror 시 pose 정렬, 관측이 있는 rollout 시작점 선택.
- 실제 DynaMapping `update()`와 무작위 pose pair 99개 직접 비교: 최대 오차 0.0.
  명령: `/home/intern/DynaRaySim/.venv/bin/python validation/check_dynamapping.py`.
- 명령: `bash scripts/run.sh -m unittest discover -s tests -v` 및
  `bash scripts/run.sh -m neural_scene smoke`.
- 전체 모델 CUDA forward/loss/backward 및 실제 500-update GPU 학습·평가 완료.
  Best validation checkpoint update 150, test F1=0.2535819, measurement F1=0.2551800.
  단기 실험이며 원 논문 metric 및 장기 학습의 재현을 주장하지 않는다.
- 실제 학습 중 한 voxel의 BatchNorm 오류를 수정했다. 해당 호출만 running 통계로
  정규화하며 affine/feature gradient와 기존 checkpoint keys를 유지한다.
  Singleton block/full U-Net backward 및 checkpoint 이후 log replay 검증 포함 30 tests 통과.
- 새로운 DynaRaySim composed surface GT archive 7개 생성, 실제 DynaMapping 전처리
  630 frames 완료. Train/validation/test = 450/90/90, 시나리오 단위 split.
  전 프레임 grid/GT voxel uniqueness/timestamp/relative pose 검증 통과,
  저장된 relative pose와 호환 계산의 최대 오차 0.0.
