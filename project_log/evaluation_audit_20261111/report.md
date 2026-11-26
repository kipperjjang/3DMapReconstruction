# evaluation.py 로컬 실행 점검 — 2026-11-11

`conda activate dyna` 후 `bash scripts/run_dyna.sh`로 실행했다.
PyTorch 2.9.1+cu130, CUDA 13.0, NVIDIA GeForce RTX 5060 Ti에서 확인했다.
샌드박스에서는 GPU 장치가 보이지 않아 GPU/GUI 실행은 승인된 외부 실행으로 진행했다.

## 수정한 문제

1. **기본 실행의 MinkowskiEngine import 실패**: `.deps-dyna`가 없고
   `.deps-dyna-cuda`만 설치된 이 컴퓨터에서 `run_dyna.sh`가
   `ModuleNotFoundError`로 실패했다. 기본 경로에 ME가 없을 때 CUDA 의존성
   디렉터리를 선택하도록 수정했다. 명시적인 `NEURAL_SCENE_DYNA_DEPS` 설정은 우선한다.
2. **잘못된 mesh 입력에서 불필요한 기록 생성**: 존재하지 않는 OBJ를 지정하면
   재생 실패 전에 `.rrd`를 생성했다. 모든 mesh를 창/기록 생성 전에 읽고 검증한다.
   유효한 mesh는 반복 재생에서도 다시 디스크에서 읽지 않는다.
3. **VisPy 조작 안내 잘림**: 실제 렌더링에서 여러 줄 안내의 마지막 줄만 보였다.
   텍스트 anchor를 수정해 조작 키와 불투명도가 모두 창 안에 표시되게 했다.

mesh 실패 처리와 안내 표시에는 회귀 테스트를 추가했다. 원래 존재하던
다른 작업의 수정 사항과 학습 가중치는 유지했다.

## 실제 가중치 평가 범위

주 모델: `outputs/training_run_v4_temporal/best.pt`

SHA256: `077777892564f8dbe7598b96667ec735798d523fd37072c91b6bcc1fc4970280`

데이터: `outputs/training_dataset_v4`

| 실행 | 범위 | 프레임 | 모델 F1 | 관측 baseline F1 |
|---|---|---:|---:|---:|
| validation | 4개 장면 × 처음 100 | 400 | 0.990809 | 0.413306 |
| test | 4개 장면 × 처음 100 | 400 | 0.993267 | 0.418767 |
| train | 32개 장면 × 처음 5 | 160 | 0.931325 | 0.402190 |
| no-feedback | validation과 동일, feedback 끔 | 400 | 0.768926 | 0.413306 |
| CPU | scene_0000 처음 2 | 2 | 0.971371 | 0.416087 |
| v3 temporal best.pt | v3 scene_0000 처음 5 | 5 | 0.863796 | 0.354962 |
| v3 legacy best.pt | v3 scene_0000 처음 5 | 5 | 0.887267 | 0.354962 |
| limit 생략 | 실제 v4 프레임으로 구성한 12프레임 점검용 sequence | 12 | 0.995340 | 0.416087 |

총 1,384프레임이다. **전체 데이터셋 성능 평가가 아닌 기능 점검**이다.
v3 두 모델은 각각 `training_run_v3_temporal/best.pt`, `training_run_v3/best.pt`를
로드했다. `--limit` 생략은 원본 HDF5를 hardlink하고 manifest의 프레임 목록만
12개로 줄인 별도 데이터셋에서 확인했다. 원본 manifest/HDF5는 수정하지 않았다.

모든 NPZ의 점군 차원·유한값·confidence 길이·pose·frame ID를 검사했다.
프레임별 TP/FP/FN/프레임 수의 합이 report.json과 일치했다.
CPU/GPU 첫 2프레임의 점군을 voxel 좌표로 정렬한 뒤 비교했을 때
좌표의 최대 절대 차이는 `2.384185791015625e-07 m`였다.

## 기능 확인

- export: checkpoint 3종, CPU/GPU, train/validation/test, 장면 선택,
  장면당 limit 및 생략, feedback 유무, 보고서/manifest/NPZ/source mesh 저장.
- play: VisPy와 실제 Rerun GUI, Rerun headless `.rrd` 저장,
  prediction/observation/target, reference target/observation/none, 세 layer 동시 표시,
  각 layer/mesh 불투명도, 점군/voxel, 자동/명시적 mesh,
  이전 export의 이동한 dataset 경로 복구, fps/max-points/point-size/distance.
- VisPy 실제 창에서 2프레임을 두 번 반복하고 Escape로 종료했다.
  마지막 화면 유지도 실제 event loop에서 timer로 창을 닫아 반환을 확인했다.
  `--no-hold`, opacity 변경/숨김/복원 및 R recenter도 확인했다.
- 잘못된 fps/opacity/headless/loop 조합, 없는 장면,
  여러 장면에 단일 mesh 지정, 기존 기록 파일 재사용은 의도대로 거부했다.
  없는/잘못된 mesh는 창과 기록을 생성하기 전에 실패한다.
- 기존 export 폴더 재사용, 미완료 export, 외부 경로 접근 거부는 회귀 테스트로 확인했다.

최종 평가/시각화 테스트 **11 passed** (skip 없음):

```bash
conda activate dyna
RUN_VIEWER_TESTS=1 bash scripts/run_dyna.sh -m pytest \
  tests/test_evaluation.py tests/test_evaluation_visualization.py -q
```

별도 `evaluation_realtime.py`의 비GUI 테스트도 통과했다(4 passed).
실시간 시뮬레이터 통합 테스트 1개는 기본 설정에 따라 skip했으며,
실제 조이스틱과 realtime GUI는 이번 `evaluation.py` 점검 범위에 포함하지 않았다.
MinkowskiEngine의 기존 autograd deprecation warning 및 Rerun의 그래픽 backend
warning은 남아 있지만 이번 실행과 렌더링은 완료됐다.

## 결과와 재생

결과는 `outputs/evaluation_audit_20261111/`에 있다.

```bash
conda activate dyna
bash scripts/run_dyna.sh -m neural_scene.evaluation play \
  --input outputs/evaluation_audit_20261111/validation \
  --layers prediction observation target --mesh
```

`commands.json`에는 22개 CLI 시나리오의 실제 명령과 예상/실제 종료 코드가 있다.
validation 최초 실행은 `validation.log`, 추가 lifecycle 검증은 `lifecycle.json`과
`lifecycle.log`, 배열·지표 검증은 `export_checks.json`, 최종 pytest 결과는
`final_tests.log`에 기록했다. `vispy.png`는 수정 후 실제 canvas 이미지다.
`missing_mesh_before_fix.rrd`는 수정 전 오류 재현에서 생성된 기록이다.

`run_audit.py`는 성공한 CLI 시나리오를 `commands.json`에서 읽어 건너뛰고
추가 점검을 이어서 수행하는 점검 스크립트다. 완전히 새로 반복하려면 기존 결과를
보존한 상태에서 스크립트의 OUT/LOG를 새 디렉터리로 바꿔 실행한다.
