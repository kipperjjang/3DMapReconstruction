# Basic-solid terrain 수집과 학습

2026-11-10에 `nohup setsid`로 호스트 백그라운드 파이프라인을 시작했다.
대화 종료와 터미널 연결 종료 후에도 실행된다. 서버 종료·재부팅 후 자동 재시작하는 서비스는 아니다.

사용자가 지정한 `basic_slc`는 존재하지 않아, 실제 경로인
`../../../DynaRaySim/resources/assets/basic_solid`의 27종 OBJ를 사용했다.
기존 검증 전용 데이터 `outputs/validation_basic_solids`는 학습에 포함하지 않는다.

| 항목 | 설정 |
|---|---|
| 데이터 규모 | Part 1과 동일: 20개 독립 장면 × 10,000프레임 = 200,000프레임 |
| Train / validation / test | 160,000 / 20,000 / 20,000프레임, 장면 단위 분리 |
| 장면 | 40 × 40m 평지, 물체 160개 |
| 물체 구성 | native basic solid 27종 × 2개 + 무작위 크기 상자 106개 |
| 기본 도형 | 원본 크기 유지, 무작위 yaw와 위치, 모든 장면에 27종 전부 포함 |
| 경로 | v4의 1,000초 roaming curve, 물체 AABB와 연속 선분 사이 0.55m 여유 |
| 관측 | 10Hz, 32채널 LiDAR, 수평 3°, robot frame의 64³ / 5cm grid |
| 수집 | CPU worker 1개, 전체 저장 상한 10GB |
| 학습 | 새 모델, CUDA, 20,000 updates, batch 4, rollout 12 |
| 학습 설정 | aligned_temporal_t01, position noise 0.005m, LR 0.01 → 0.0001, seed 20261109 |
| 수집 seed | 20261110 + scene ID |

기본 도형의 배치는 전체 경로에서 중심까지 1.55m 이내로 제한한다.
시뮬레이션 전에 20개 장면 모두 geometry 생성·물체 간 겹침·경로 여유 검사를 통과했다.

파일럿은 전체 크기 장면에서 120프레임을 새로 수집했다. 전체 shard 검증과 독립 재현
3프레임 검사를 통과했다. CUDA 모델 smoke 검사도 실제 GPU forward/backward를 통과했다.
파일럿은 본 데이터의 200,000프레임에 포함하지 않는다.

본 수집 후 전체 데이터 검증과 60프레임 독립 audit를 통과해야 학습을 시작한다.
새 학습은 기존 학습이 실행 중이어도 동시에 시작한다.
수집·학습의 소스 사본, 원본 OBJ 사본과 SHA256을 저장한다.
실패는 상태 파일과 로그에 기록하며, 검증 실패 시 학습을 시작하지 않는다.

## 상태와 결과

```bash
python3 scripts/status_dataset_v4.py --dataset outputs/training_dataset_v4_solids
cat project_log/dataset_solids/pipeline/status.json
tail -f project_log/dataset_solids/production/scene_0000.log
tail -f project_log/dataset_solids/pipeline/training.log
```

| 경로 | 내용 |
|---|---|
| `../../outputs/training_dataset_v4_solids` | 본 데이터·진행 상태·검증 보고서 |
| `../../outputs/training_run_v4_solids` | 학습 로그·TensorBoard·checkpoint·최종 report.json |
| `../../project_log/dataset_solids/pipeline/status.json` | 전체 파이프라인 phase·PID·heartbeat·실패 원인 |
| `../../project_log/dataset_solids/production` | 각 장면·전체 검증·독립 audit 로그 |
| `../../project_log/dataset_solids/pipeline/source` | 실행 시작 시 고정한 수집·학습 코드 |
| `../../outputs/dataset_solids_pilot_v3` | 성공한 120프레임 파일럿과 검증 결과 |

`phase=collecting_and_verifying`는 수집과 검증을 진행하는 상태이며,
`phase=training`은 학습 중, `phase=complete`는 학습 결과 보고서가 생성된 상태다.
`phase=reverifying_completed_data`는 수집이 끝난 데이터를 재검증하는 복구 단계다.
장면의 dense GT는 프레임 수집 전에 생성하므로 초기 프레임 수 0은 정상이다.

중단하려면 `touch project_log/dataset_solids/pipeline/stop.requested`를 실행한다.
전체 파이프라인이 자식 프로세스 그룹을 종료한다. 부분 shard와 checkpoint는 남긴다.
정상 종료된 scene은 재실행 시 재사용하고, 학습은 `latest.pt`에서 resume한다.
미완료 `.partial` shard 자동 이어 쓰기는 지원하지 않으므로 중단된 장면은 점검이 필요하다.

중복 실행하지 않는다. 전체 파이프라인과 수집 supervisor는 각각 파일 잠금을 사용한다.
실행이 종료된 뒤 재시작할 때만 다음을 사용한다.

```bash
bash scripts/start_solids_pipeline.sh
```

## 2026-11-11 완료 데이터 복구

20개 장면 200,000프레임 수집과 전체 검증·60프레임 audit는 완료됐지만,
수집 중 `DynaMapping/dynamapping/viewer.py`가 변경되어 마지막 provenance 검사가 실패했다.
수집에서 import하지 않는 단독 viewer의 변경만 예외로 기록하며,
시뮬레이터·mapping core·고정한 수집 코드·원본 asset·geometry·scenario 변경은 허용하지 않는다.
원본 `provenance.json`은 수정하지 않는다.

`../../scripts/finalize_solids_dataset.py`는 기존 20만 프레임을 다시 전체 검증하고
60프레임 독립 audit를 재실행한 뒤 `result.json`을 생성한다.
검증 근거와 변경 전후 SHA256은 `project_log/dataset_solids/production/recovery_evidence.json`에 남긴다.
다음 명령은 기존 파이프라인이 종료된 상태에서 해당 viewer 변경으로 막힌 완료 데이터를 복구할 때 사용한다.

```bash
bash scripts/start_solids_pipeline.sh --finalize-existing
```

복구를 통과하면 수집을 반복하지 않고 학습을 자동 시작한다.
