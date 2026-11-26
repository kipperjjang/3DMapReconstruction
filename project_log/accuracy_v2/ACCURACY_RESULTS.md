# 정확도 개선 실험 — 2026-10-12

선택한 새 모델은 **학습 관측 위치 노이즈를 ±5cm에서 ±5mm로 낮춘 모델**이다.
동일한 90-frame 테스트 시나리오에서 F1이 **0.253582 → 0.313151**로 향상됐다.
이는 **+0.059569, 상대 +23.49%**이며, precision 향상과 recall 감소를 함께 보였다.

## 기존 결과와 비교

| 방법 | Precision | Recall | F1 | FP voxel |
|---|---:|---:|---:|---:|
| 현재 관측만 voxelization | 0.948681 | 0.147416 | 0.255180 | 4,376 |
| 기존 학습 모델 v1 | 0.207492 | 0.325994 | 0.253582 | 683,263 |
| 새 모델 `noise005` | **0.345593** | 0.286277 | **0.313151** | **297,471** |

오검출 voxel은 56.46% 줄었다. Precision은 13.81%p 높아졌고 recall은 3.97%p 낮아졌다.
GT와 일치하는 voxel에서의 centroid distance는 2.127cm에서 1.807cm로 감소했다.
이 거리 오차는 matched subset만 측정하며 누락/오검출 오차를 포함하지 않는다.

## 실험 과정과 채택하지 않은 후보

| 실험 | 변경 | 학습 업데이트 | Best step | Validation F1 | Test F1 |
|---|---|---:|---:|---:|---:|
| baseline_v1 | 기존 설정 | 500 | 150 | 0.169148 | 0.253582 |
| **noise005** | 위치 노이즈 ±5mm | 500 | 100 | **0.205818** | **0.313151** |
| noise005_lr001 | 위 설정 + 초기 LR 0.001 | 500 | 100 | 0.175797 | 평가 보류 |
| best_train_bncal | noise005 가중치 고정, clean train 32 rollouts로 BN 통계 재추정 | 0 | 통계 재추정 | 0.193656 | 평가 보류 |

초기 LR 감소와 BN 통계 재추정은 noise005보다 검증 성능이 낮아 채택하지 않았다.
낮은 성능의 실험도 가중치, 설정, 소스 사본, 로그, 보고서를 보존했다.

학습 장면에서 매 5번째 frame을 표본으로 삼은 총 90 frames의 진단 결과,
관측 occupied voxel이 GT와 일치하는 비율은 노이즈 없이 90.67%, ±5mm에서 74.52%,
±5cm에서 50.98%였다. 이 값은 augmentation이 관측 support에 미치는 영향이며,
네트워크 정확도나 모든 오검출의 원인을 증명하는 수치는 아니다.

## 비교 조건

- 기존 dataset/split 그대로: train 450, validation 90, test 90 frames.
- 기존 t=0 점군 병합, DynaMapping 상대 pose 정렬 및 detached prediction feedback.
- 동일 채널 `[32,64,128,256,256]`, 단계당 일반 conv 2개, 64³ grid, voxel 5cm.
- 학습·추론 predictive pruning alpha=0.5, GT support 강제 유지 없음.
- 새 학습 실험 모두 batch 2, rollout 12, seed 20261011, 500 optimizer updates.
- 선택 모델의 Adam LR 0.01→0.0001 및 mirror 설정은 v1과 같다.
- 위치 노이즈 감소는 원 논문 augmentation 수치에서의 명시적인 실험적 변경이다.
- 후보 및 checkpoint 선택은 validation F1만 사용했다. 선택과 checkpoint SHA256을
  `selection.json`에 고정한 뒤 선택 모델에 대해서만 test를 평가했다.
- 기존에 평가한 같은 test 시나리오를 사용한 반복 benchmark이다. 새로운 미지 데이터나
  여러 seed/환경에 대한 일반화 성능을 입증한 결과로 해석하지 않는다.

## 파일과 재현

- 새 모델: `outputs/accuracy_study_v2/runs/noise005/best.pt`
- 최종 평가: `outputs/accuracy_study_v2/final_evaluation/report.json`
- 순차 추론 예시: `outputs/accuracy_study_v2/reconstruction/frame_00000.npz`, `frame_00001.npz`.
  두 번째 frame은 첫 예측을 현재 robot frame으로 정렬해 병합한다.
- 전체 비교: `outputs/accuracy_study_v2/comparison.csv`, `comparison.json`, `COMPARISON.md`
- 변경 없이 누적하는 기록: `outputs/accuracy_study_v2/experiments.jsonl`
- 선택 및 테스트 연결: `selection.json`, `selected_test.json`
- 원본 v1 보존 확인: `baseline_artifact_hashes.json`의 20개 파일 SHA256 모두 일치.
- 각 새 실험: `configuration.json`, `console.log`, `source/`, `source_hashes.json`과 결과 파일.
- 이전 보고서 [TRAINING_RESULTS.md](../../docs/TRAINING_RESULTS.md)와 v1 checkpoint는 그대로 보존했다.

선택 모델의 학습 재현은 **새 output 경로**로 실행한다.

```bash
bash scripts/train_gpu.sh --output outputs/noise005_reproduction \
  --position-noise 0.005 --skip-test
```

GPU sparse 연산의 비결정성이 있어 같은 seed도 bit 단위 재현을 보장하지 않는다.
비교표는 `bash scripts/run.sh scripts/compare_experiments.py`로 다시 생성할 수 있다.
이미 기록된 보고서의 hash가 달라지면 기록을 중단하여 과거 결과 변경을 알린다.

검증은 32개 테스트를 통과했다. 후보 학습에서 test 평가가 호출되지 않는지,
BN 재추정이 train frame만 읽고 학습된 parameter를 변경하지 않는지,
선택한 checkpoint와 실제 평가 checkpoint hash가 일치하는지도 확인했다.
