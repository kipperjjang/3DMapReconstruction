# 넓은 평지·상자 지형 데이터셋 v4

v3의 좁은 통로 왕복 경로를 넓은 평지의 연속 이동·회전 경로로 바꾼 수집 작업이다.
`/home/intern/DynaRaySim/torch`, `/home/intern/DynaMapping/torch`, conda `dyna`를 사용한다.
기존 데이터셋과 학습 결과는 보존하고 새 데이터는 `../../outputs/training_dataset_v4`에 저장한다.

## 구성과 목표

| 항목 | v3 | v4 목표 |
|---|---|---|
| 지형 크기 | 10×8m | **40×40m**, 면적 20배 |
| 지형 형태 | 박스·계단·벽·통로 | 깊이 2.5m의 평평한 지면과 **정적 상자 160개** |
| 상자 크기 | 제한된 범위 | 가로·세로 각각 0.2–3m log-uniform, 높이 0.1–2.2m uniform |
| 이동 | 한 통로에서 왕복 | 여러 방향으로 이동하는 매끄러운 곡선, 진행 방향을 따라 yaw 변화 |
| 속도 | 경로에 따라 변화 | 장면별 약 0.65–0.85m/s, 경로 길이 기준으로 균등 샘플 |
| 장면별 duration | 500초 | **1,000초**(약 16.7분) |
| 수집 주기 | 10Hz | 10Hz |
| 장면별 프레임 | 5,000 | **10,000** |
| 장면 수 | 40 | 40 |
| 총 프레임 | 200,000 | **400,000** |
| Train / validation / test | 160k / 20k / 20k | **320k / 40k / 40k**, 장면 단위 분리 |
| 새 데이터 용량 상한 | 10GB | **10,000,000,000 bytes**, 512MiB 여유를 두고 중단 |

총 시뮬레이션 시간은 40,000초(약 11.1시간)이며 실제 수집에 걸리는 시간과 다르다.
프레임 수는 고유 수집 시각 수이며 point 수가 아니다. 파일럿은 본 수집 수량에 포함하지 않는다.
상자는 움직이지 않으며 현재 버전에는 dynamic object가 없다.

## 경로와 센서

경로는 여러 제어점을 잇는 Catmull–Rom 곡선을 길이 기준으로 샘플링한다.
고정된 두 끝점을 반복 왕복하거나 제자리에서 회전하는 경로를 사용하지 않는다.
Yaw는 경로의 접선 방향을 따르며 roll/pitch는 0이다.
40개 예정 장면 모두 X·Y 방향 각각 20m 이상을 탐색하고 정지 segment가 없는 것을 확인했다.
각 scene의 경로는 결정적인 seed로 재현할 수 있다.

상자는 경로 생성 후 배치한다. 모든 **연속 선분**과 각 상자의 XY footprint를
검사하고, 상자와 경로 사이에 보수적인 0.55m 여유가 있는 배치만 허용한다.
상자끼리도 겹치지 않게 배치한다. Simulator의 motion rejection과 기록된 pose 중복도 검사한다.

LiDAR는 32채널, 수평 360°/3° 간격, 수직 −85°~45°, range 0.05–40m다.
매 프레임 LiDAR Z축과 world Z축의 일치 및 실제 DynaMapping 상대 변환을 검사한다.

**넓어지는 것은 수집 world와 이동 범위다.** 네트워크 입력은 기존과 같은
현재 robot frame의 3.2×3.2×3.2m, 64³/5cm grid다.
현재 관측·GT·pose를 분리 저장하므로 기존 reader와 시간 구분 모델에 사용할 수 있다.

GT는 현재 DynaRaySim의 composed boundary 알고리즘을 사용하고, 경계 5cm·표면 샘플
최대 edge 3cm를 유지한다. 넓은 geometry의 GT를 장면당 한 번 만든 후 pose에 따라 crop한다.
가시 영역만으로 GT를 제한하거나 논문의 43%에 맞춰 frame을 골라내지 않는다.
`gt_point_coverage_5cm`와 strict `gt_voxel_recall`은 별도로 기록한다.
지면 아래쪽 면이 학습 cube에 들어오지 않도록 floor의 바닥은 z=−2.5m다.

## 백그라운드 실행 흐름

현재 작업은 장시간 실행 가능한 관리 터미널에서 실행한다. 수집 프로그램은 LLM의
추가 호출 없이 아래 순서를 실행하고 상태·결과를 파일로 남긴다.

1. 전체 크기 지형의 120프레임 파일럿을 수집한다.
2. 파일럿 전체 shard 검증과 대표 3프레임의 독립 재현 검사를 수행한다.
3. 파일럿 성공을 기다리던 production supervisor가 4개 worker로 본 수집을 시작한다.
4. 전체 400,000프레임을 검증한다.
5. 각 scene의 첫·중간·마지막 프레임, 총 120프레임을 원본 mesh에서 다시 ray casting하고
   공유 dense GT 전체를 독립적으로 crop해 저장 결과와 비교한다.
6. 성공 시 `result.json`, `status.json`, `RESULTS.md`를 기록한다.

파일럿이나 worker가 실패하면 본 수집을 시작하거나 성공으로 표시하지 않으며,
`status.json`에 `phase=failed`와 원인을 남긴다. 한 데이터 폴더에 supervisor가
중복 실행되지 않도록 파일 잠금을 사용한다. Worker·검증 코드는 로그 폴더의
`source/` 사본으로 고정하고, simulator/mapper 소스 hash도 수집 전후 비교한다.

이 실행은 모델 응답이 끝나도 추가 모델 호출을 요구하지 않는다. 서버 재부팅이나
실행 환경 자체의 종료를 넘어 지속되는 서비스는 아니다.

호스트 터미널에서 별도로 시작해야 할 경우, 먼저 현재 상태와 중복 실행 여부를 확인한 뒤:

```bash
cd "/home/intern/Neural Scene unet"
conda activate dyna
bash scripts/start_dataset_v4.sh --pilot-dataset outputs/dataset_v4_pilot_final
```

이 명령의 기본 동작은 호스트의 `nohup` 백그라운드 실행이다.
`--foreground`는 관리 터미널에서 프로세스를 계속 실행하는 모드다.
수집 중인 작업을 두 번 시작할 필요는 없다.

## 진행 상황 확인

추가 의존성 없이 상태를 확인할 수 있다.

```bash
cd "/home/intern/Neural Scene unet"
python scripts/status_dataset_v4.py

# 파일럿 진행 상황
python scripts/status_dataset_v4.py --dataset outputs/dataset_v4_pilot_final

# Supervisor 로그
tail -f project_log/dataset_v4/production/supervisor.log

# 개별 production scene 로그 (GT 생성 중에는 새 frame 로그가 없을 수 있음)
tail -f project_log/dataset_v4/production/scene_0000.log
```

| 상태 항목 | 의미 |
|---|---|
| `phase=waiting_for_pilot` | 파일럿의 수집·검증 완료를 기다리는 중. 아직 본 수집을 시작한 상태가 아님 |
| `phase=collecting` | Geometry/GT 준비 또는 프레임 수집 중 |
| `phase=validating` | 전체 frame/hash/pose/GT voxel 검사 중 |
| `phase=auditing` | 대표 프레임의 독립 raycast·GT 재구성 검사 중 |
| `phase=ready_for_training` | 모든 수집·검증 완료 |
| `phase=failed` | 중단 또는 실패. `error`와 해당 stage/scene 로그 확인 |
| `acquired_frames_lower_bound` | 100프레임마다 flush되는 로그 기준의 보수적 하한 |
| `completed_scenes` | 정상적으로 닫힌 HDF5와 metadata를 갖춘 scene 수 |
| `dataset_bytes` | 해당 새 데이터 폴더의 논리적/할당된 용량 중 큰 값의 합 |
| `heartbeat_stale=true` | 최종 완료·실패 상태가 아닌데 heartbeat가 90초 넘게 갱신되지 않음 |

넓은 지형의 dense GT 생성은 첫 frame보다 먼저 수행되어 초기에는 프레임 수가 0일 수 있다.
상태 heartbeat가 계속 갱신되는지와 scene 로그를 함께 본다.

## 파일 위치와 완료 확인

| 경로 | 내용 |
|---|---|
| `../../outputs/training_dataset_v4/status.json` | phase·heartbeat·진행량·실패 원인 |
| `../../outputs/training_dataset_v4/collection_settings.json` | 수집 목표·geometry·센서·seed 설정 |
| `../../outputs/training_dataset_v4/provenance.json` | 실제 환경과 simulator/mapper 및 pipeline 소스 hash |
| `outputs/training_dataset_v4/sequences/scene_XXXX/` | mesh, scenario, terrain/GT metadata, dense GT, frame shard |
| 각 scene의 `terrain_metadata.json` | 상자 위치·크기, 경로 범위·길이·속도·충돌 여유 |
| `../../outputs/training_dataset_v4/dataset.json` | 본 수집이 모두 끝나면 생성되는 학습 manifest |
| `../../outputs/training_dataset_v4/validation_report.json` | 전체 frame 수·shard hash·pose·축·GT 검사 결과 |
| `../../outputs/training_dataset_v4/sample_audit.json` | 독립 재현 검사 결과 |
| `../../outputs/training_dataset_v4/result.json` | 모든 검사 통과 후 최종 요약 |
| `../../project_log/dataset_v4/production/RESULTS.md` | 완료 후 사람이 읽을 수 있는 결과 요약 |
| `project_log/dataset_v4/production/*.log` | supervisor, 각 scene, validation, audit 로그 |
| `../../project_log/dataset_v4/terrain_preview.svg` | 별도 파일럿 seed의 지형·장기 경로를 나타낸 평면도 |

완료 판단은 `dataset.json`의 존재만으로 하지 않는다. `status.json`과 `result.json`의
상태가 `ready_for_training`이고, 검증 frame 수 400,000 및 audit 120개가 통과해야 한다.
`result.json`의 크기는 마지막 요약 파일 작성 직전 값이므로 최종 디렉터리 크기와 소량 차이날 수 있다.

프레임 저장은 v3와 같은 6-byte quantized point + gzip이며 약 0.17mm의 3D 오차 상한을 갖는다.
관측 중복 point를 보존한다. 이 상한과 float 반올림을 포함하여 audit 허용 거리는 0.18mm다.

## 중단과 재실행

명시적으로 중단하려면:

```bash
touch outputs/training_dataset_v4/stop.requested
```

Supervisor가 worker를 종료하고 partial shard를 보존한다.
닫힌 scene은 같은 설정·소스의 재실행에서 재사용할 수 있지만,
**중단된 scene의 `.partial` 파일 자동 이어 쓰기는 지원하지 않는다.**
실패 원인과 남은 파일을 확인한 뒤 새 수집 폴더를 사용하거나 해당 미완료 scene을 별도로 정리해야 한다.
`stop.requested`가 있으면 다시 실행해도 중단되므로 재시작 전에 해당 요청을 해제해야 한다.

## 20만 프레임씩 분할하여 학습

완료된 v4 원본 40만 프레임을 장면 단위로 두 묶음으로 나눈다.
같은 장면을 잘라 양쪽에 넣지 않으므로 장면·frame 중복이 없으며,
각 장면의 1,000초 시계열과 기존 train/validation/test 소속을 유지한다.

| 묶음 | 경로 | 장면 | Train / validation / test | 합계 |
|---|---|---|---|---|
| Part 1, 기본 학습 대상 | `../../outputs/training_dataset_v4_part1` | scene_0000–0019 | 160,000 / 20,000 / 20,000 | 200,000 |
| Part 2 | `../../outputs/training_dataset_v4_part2` | scene_0020–0039 | 160,000 / 20,000 / 20,000 | 200,000 |

분할 생성 명령은 다음과 같다. 이미 생성된 출력 폴더는 덮어쓰지 않는다.

```bash
bash scripts/run_dyna.sh scripts/split_dataset_v4.py
```

HDF5·dense GT·mesh 등의 불변 파일은 **hard link**로 연결하여 대용량 파일을
복제하지 않는다. 각 폴더 안에는 실제 파일 경로가 있으므로 reader의 경로 검사도 통과한다.
Hard link는 같은 파일 시스템에서 사용하며, 공유한 파일을 수정하면 원본에서도 바뀌므로
공유 데이터 파일은 읽기 전용으로 취급한다. Manifest, scenario 및 sequence metadata는
각 묶음에 별도로 저장하고 scenario의 mesh 경로와 config hash도 맞춰 갱신한다.

각 묶음의 `partition.json`에 원본 manifest SHA256과 소속 scene을 기록한다.
`validation_report.json`은 분할 후 20만 프레임 전체 검증 결과다.
`sample_audit.json`은 데이터·geometry·pose가 동일한 원본의 독립 재현 검사 중
해당 20개 장면의 60개 샘플을 선택한 것으로, 새 ray casting을 수행했다는 뜻은 아니다.
이 상속 관계는 파일의 `inheritance`에 명시한다.

`train_v4.sh`의 기본 데이터는 **Part 1**, 기본 출력은
`../../outputs/training_run_v4_part1`이다. 학습 설정은 기존 v3_temporal과 동일하다.
목표 프레임 수는 선택한 묶음의 `collection_settings.json`에서 읽는다.

```bash
conda activate dyna
bash scripts/train_v4.sh
```

학습 길이는 `--epochs 10` 또는 `--updates 20000`으로 지정한다(동시 사용 불가).
예: `bash scripts/train_v4.sh --epochs 10 --no-tensorboard`.
현재 무작위 rollout 샘플링을 유지하므로 epoch는 전체 train 프레임 수에 해당하는
처리량이다. 모든 프레임을 한 번씩 순회한다는 뜻은 아니다.
epoch당 update는 `ceil(train_frames / (batch_size * rollout))`이며,
Part 1의 batch 2, rollout 12에서는 6,667 updates다.
설정과 로그에 epoch를 기록하고, LR decay도 계산된 총 update 수를 따른다.
`--resume`할 때는 최초 실행과 같은 epoch 수를 지정한다.

Part 2를 따로 학습하려면 데이터와 출력 폴더를 함께 지정한다.

```bash
bash scripts/train_v4.sh \
  --dataset outputs/training_dataset_v4_part2 \
  --output outputs/training_run_v4_part2
```

20,000 updates × batch 2 × rollout 12의 480,000 frame presentations는 각 묶음의
train 160,000프레임을 약 세 번 사용하는 양으로, v3의 데이터 사용량과 대응한다.
기존 40만 프레임 run의 checkpoint를 다른 묶음으로 `--resume`하면 데이터 hash가
달라 거부된다. 각 묶음의 새 run 폴더에서 시작한다.
원본 수집 완료 보고서의 400,000프레임과 분할 데이터의 200,000프레임을 구분한다.
