# 최신 GT 확장 데이터셋 — 2026-10-20

데이터 생성·정렬·검증은 완료했다. 전체 새 데이터셋은 **804,196,352 bytes
(0.804 GB)**이며, 원시 DynaRaySim archive와 학습 NPZ를 모두 포함해 5 GB 미만이다.
본 GPU 학습은 실행 환경에 NVIDIA 장치가 노출되지 않아 시작하지 못했다.

| 구분 | 기존 v1 | 새 v2 |
|---|---:|---:|
| Train frames | 450 | 1,040 |
| Validation frames | 90 | 180 |
| Test frames | 90 | 200 |
| 전체 frames | 630 | 1,420 |
| 관측이 있는 12-frame 학습 시작점 | 332 | 769 |

7개 시나리오의 전체 구간을 최신 소스로 재수집했다. 기존과 동일한 시나리오별 split을
유지했으며, 기존 데이터·모델을 보존했다. 관측이 없는 train frame 216개는 이전 예측의
feedback 처리를 위해 유지하고, 그 프레임에서 새로운 rollout을 시작하지 않는다.

## GT와 정렬

- DynaRaySim 최신 PyTorch 구현의 `composed` surface GT: 경계 격자 5cm,
  삼각형 최대 edge 3cm. 저장량이 작지만 동적 접촉면과 공간 개폐에 대한 근사가 남는다.
- 과거 auto 검증 보고서는 최신 ray-pattern fingerprint와 달라 사용하지 않았다.
  숫자로 지정한 해상도를 사용했으며, 현재 코드에 대한 경계 해상도 감사 완료를 주장하지 않는다.
- 실제 `/home/intern/DynaMapping`을 사용했다. 관측을 기록된 sensor pose와 robot pose로
  robot frame에 옮긴 후 DynaMapping에서 frame alignment를 계산했다.
- 현재 관측과 이전 네트워크 예측을 정렬·병합한 뒤 voxelization하는 기존 입력 정책과
  전체 네트워크 구조를 유지한다. GT는 robot frame의 64³, 5cm grid 내부 표면 centroid다.

## 검증 결과

- 1,420개 전체 파일의 SHA256, 유한 좌표, grid 범위, GT voxel 중복, timestamp,
  frame 간 pose 변환을 검사했다. 독립적으로 계산한 상대 변환과의 최대 차이는 0이다.
- 시나리오별 처음/중간/마지막의 21프레임을 NumPy 강체 역변환으로 재구성해 저장된
  관측·GT와 비교했다. 모두 통과했다. 이 중 관측이 있는 19프레임에서는 모든 관측점이
  GT 표면의 5cm 이내였고 최대 거리는 2.72cm였다. 대표 시각·센서 근처의 검사이며,
  전체 GT 경계의 정확도 보장은 아니다.
- 초기 감사에서 일반 행렬 역산의 약 1e-15 반올림 차이가 grid 경계 벽의 voxel 배정을
  바꿨다. 강체 pose 규약에 맞는 `R.T`, `-R.T @ t`로 독립 비교를 수정했다.
  데이터와 허용 오차는 변경하지 않았고 초기 검사 로그도 보존했다.
- 기존 네트워크 테스트 32개, 새 DynaMapping/용량 제한 테스트 2개가 통과했다.
- 실제 새 데이터로 전체 채널 모델의 batch 2 / rollout 12, Adam 업데이트 **1회**를
  CPU에서 완료했다. Loss/gradient 유한성 및 가중치 변경을 확인했다.
  평균 loss 1.2473, 약 69.9초. 이는 실행 검사이며 학습 완료 모델이나 성능 평가가 아니다.

## 본 학습 상태와 실행

GPU 사전 검사: `/dev/nvidia*` 없음, `torch.cuda.is_available=False`.
따라서 본 학습의 optimizer update는 **0회**, 새 best checkpoint 및 test 성능은 없다.
CPU 측정값을 단순 환산하면 1,500회에 약 29시간이며 validation 시간은 별도다.

GPU가 노출된 호스트에서 이 프로젝트 폴더를 열고 실행한다.

```bash
bash scripts/train_v2.sh
# 동일 run 중단 후 재개
bash scripts/train_v2.sh --resume
```

준비된 설정: 1,500 updates, batch 2, rollout 12, ±5mm 관측 노이즈, XY mirror,
Adam LR 0.01→0.0001, seed 20261020, 100회마다 validation. Validation F1으로 best를
선택한 후 test를 평가한다. GT와 평가 구간이 달라 과거 90-frame test 수치와 직접
비교하지 않는다. `outputs/training_run_v2/report.json`의 `status=completed`로
본 학습·평가 완료를 확인한다.

데이터: `outputs/training_dataset_v2/`. 전체 검증은 `validation_report.json`,
대표 프레임 검사는 `sample_audit.json`, 원본 코드 해시는 `provenance.json`에 있다.
실행 상태·코드 사본·초기 실패를 포함한 로그는
[project_log/dataset_v2]()에 보존했다.
