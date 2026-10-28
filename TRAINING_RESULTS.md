# GPU 학습 결과

RTX 4090 / CONDA `sp_conv` / PyTorch 1.13.1+cu117 / MinkowskiEngine 0.5.4 CUDA backend에서
실제 데이터 학습·평가를 완료했다. 결과는 `outputs/training_run_v1/report.json`에 기록했다.

## 데이터와 설정

- 새로운 DynaRaySim composed surface GT archive 7개, 실제 DynaMapping 전처리 630 frames.
- 시나리오 단위 분리: train 450 / validation 90 / test 90 frames.
- 이전 예측을 현재 robot frame으로 정렬하고 현재 관측과 합친 뒤 voxelization, 입력 t=0.
- 채널 `[32,64,128,256,256]`, 단계당 일반 conv 2개, 47,537,383 parameters.
- 500 optimizer updates, batch 2, rollout 12, detached feedback.
- Adam exponential LR 0.01 → 0.0001, position noise ±0.05m, rollout 전체 XY reflection.
- Training/inference pruning threshold 0.5, GT support 강제 보존 없음.
- Validation F1으로 checkpoint 선택: update 150, validation F1=0.169148.
- 원 논문의 전체 데이터 규모 및 학습 성능을 재현한 실험은 아니다.

## 별도 테스트셋 결과

`05_outdoor_obstacle_passage` 90 frames의 micro voxel metrics다.

| 방법 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 관측 점군 voxelization | 0.948681 | 0.147416 | 0.255180 |
| 학습 모델 + 정렬된 이전 예측 | 0.207492 | 0.325994 | 0.253582 |

모델 TP/FP/FN은 178,890 / 683,263 / 369,862다. 복원 coverage가 늘었지만 false positives가
많아 baseline F1을 넘지 못했다. Validation F1도 관측 baseline 0.188644보다 낮다.
학습 완료와 충분한 reconstruction 정확도 확보는 구분해야 한다.

GT와 동일한 voxel에서의 mean centroid distance는 2.127cm, height MAE는 1.028cm다.
이 수치는 matched subset만 측정하며, 누락 및 false positives의 공간 오차를 포함하지 않는다.

## 실행 중 수정과 검증

Update 456에서 downsample의 support가 한 row로 줄어 BatchNorm 오류가 발생했다.
해당 singleton 호출에만 기존 running 통계를 사용하도록 수정했다. Affine/feature gradient와
기존 checkpoint 형식은 유지하며, support나 pruning threshold는 바꾸지 않았다.
전체 30 tests를 통과한 뒤 update 450의 model/optimizer/scheduler/RNG checkpoint에서 재개했다.
재개 시 checkpoint 이후 log rows는 제거하여 업데이트 중복을 방지했다.

전체 모델 GPU forward/loss/backward, 최종 checkpoint 저장, 별도 test evaluation이 통과했다.

## 산출물

- `outputs/training_run_v1/best.pt`: validation 기준 선택 모델, update 150.
- `outputs/training_run_v1/last.pt`: update 500 모델.
- `outputs/training_run_v1/latest.pt`: optimizer/scheduler/RNG 포함 update 500 checkpoint.
- `outputs/training_run_v1/report.json`: 완료 상태, 설정, validation/test 및 baseline metrics.
- `outputs/training_run_v1/training.jsonl`: 500개의 committed update 기록.
- `outputs/gpu_training_console.log`: GPU 검사, 실행 오류 및 재개를 포함한 전체 콘솔 로그.
- `outputs/training_run_v1/reconstruction/frame_00000.npz`, `frame_00001.npz`:
  학습된 best 모델의 실제 첫 두 test frames 추론 예시. 두 번째는 첫 번째 예측을
  DynaMapping transform으로 정렬해 feedback으로 사용했다. 각각 1,560 / 1,684 points.

기존 run은 완료 상태이므로 새 실험은 다른 `--output` 경로를 사용한다.
