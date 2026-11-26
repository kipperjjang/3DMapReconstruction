# 직접 조작하며 실시간 inference 실행

`neural_scene.evaluation_realtime`은 DynaRaySim의 로봇을 키보드나 조이스틱으로
움직이며 매 센서 주기마다 새 관측을 raycast하고 학습된 모델로 추론한다.
VisPy 또는 Rerun에 prediction(청록), 관측(녹색), 장면 mesh(회색)를 표시한다.

저장된 결과를 재생하려면 [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)의
`evaluation play`를 사용한다. 실시간 모드에는 GT 점군·F1 계산·결과 export·Rerun
기록 옵션이 없다. 정량 평가는 `evaluation export`에서 수행한다.

## 1. 환경과 시나리오 준비

현재 컴퓨터에서는 프로젝트 루트에서 시작한다.

```bash
cd /home/hoon/ResearchProjects/3DMapReconstruction
conda activate dyna
bash scripts/run_dyna.sh -m neural_scene.evaluation_realtime --help
```

`run_dyna.sh`가 프로젝트 의존성과 CUDA 라이브러리 경로를 설정한다.
이 컴퓨터의 `.deps-dyna-cuda`도 자동으로 선택한다.
DynaRaySim/DynaMapping은 기본적으로 프로젝트의 형제 디렉터리에서 찾는다.
다른 위치에 설치했다면 `DYNARAYSIM_ROOT`, `DYNAMAPPING_ROOT`를 지정한다.
VisPy/GLFW가 사용할 데스크톱 디스플레이가 필요하며 headless 옵션은 없다.

**현재 v4 시나리오에는 이전 컴퓨터의 `/home/intern/...` asset 경로가 남아 있다.**
아래 명령으로 현재 장면 폴더의 asset을 가리키는 사본 2개를 만든다.
학습 데이터의 원본 YAML은 유지한다. 같은 이름의 사본은 다시 실행하면 갱신한다.
현재 DynaRaySim이 받지 않는 이전 옵션 `ego.terrain_alignment: false`도 사본에서 제거한다.

```bash
python - <<'PY'
from pathlib import Path
import yaml

root = Path.cwd()
destination = root / 'outputs/realtime_scenarios'
destination.mkdir(parents=True, exist_ok=True)
for name in ('scene_0000', 'scene_0010'):
    source = root / 'outputs/training_dataset_v4/sequences' / name / 'scenario.yaml'
    data = yaml.safe_load(source.read_text())
    data['run']['environments'] = 1
    if data['ego'].pop('terrain_alignment', False):
        raise ValueError('terrain_alignment: true requires a separate scenario migration')
    for asset in data['assets'].values():
        path = Path(asset['path']).expanduser()
        candidate = path if path.is_absolute() else source.parent / path
        if not candidate.is_file():
            candidate = source.parent / path.name
        if not candidate.is_file():
            raise FileNotFoundError(f'{source}: asset not found: {path}')
        asset['path'] = str(candidate.resolve())
    target = destination / f'{name}.yaml'
    target.write_text(yaml.safe_dump(data, sort_keys=False))
    print(target)
PY
```

사용자 정의 시나리오는 YAML당 환경이 하나여야 하며 `ego.per_environment` 설정은
지원하지 않는다. asset뿐 아니라 센서 pattern 등 YAML에 참조된 외부 파일이 있으면
사본 위치에서도 찾을 수 있도록 경로를 맞춘다. 위 예제의 v4 LiDAR pattern은 YAML
내 설정을 사용한다.

## 2. 실행

현재 존재하는 v4 part1 모델과 첫 장면을 사용한다.

```bash
bash scripts/run_dyna.sh -m neural_scene.evaluation_realtime \
  --checkpoint outputs/training_run_v4_part1/best.pt \
  --scenarios outputs/realtime_scenarios/scene_0000.yaml \
  --device cuda:0
```

Rerun을 사용하려면 같은 명령에 `--viewer rerun`을 추가한다.
현재 컴퓨터에는 경로 보정된 시나리오 사본이 이미 있으므로 다음 명령으로 실행할 수 있다.

```bash
bash scripts/run_dyna.sh -m neural_scene.evaluation_realtime \
  --checkpoint outputs/training_run_v4_part1/best.pt \
  --scenarios outputs/realtime_scenarios/scene_0000.yaml \
              outputs/realtime_scenarios/scene_0010.yaml \
  --device cuda:0 --viewer rerun
```

Rerun에서는 별도로 열리는 **Neural Scene controls** 창을 클릭한 상태에서 로봇을
조작한다. Rerun 창의 키 입력은 로봇에 전달되지 않는다. 조작 창의 Esc 또는 창 닫기로
추론을 종료한다. Rerun의 타임라인과 카메라는 Rerun 창에서 조작한다.
표의 불투명도 및 R 카메라 단축키는 VisPy 전용이며 Rerun에서는 viewer UI를 사용한다.

직접 조작 모드의 카메라 중심은 ego 원점이다. 관측·예측·mesh가 매 프레임 현재
ego 좌표계로 표시되므로 이동·회전 중에도 ego를 중심으로 관찰한다. Rerun은
`current_sensor/origin`을 추적하며 초기 카메라 거리는 `--distance`로 지정한다.
마우스로 회전·확대할 수 있다. Rerun에서 pan 등으로 추적을 해제했다면 origin을
다시 추적하거나 실행을 재시작한다.

Rerun 카메라 설정에는 `rerun-sdk==0.27.3`을 사용한다. 이 컴퓨터에는 프로젝트의
`.deps-rerun`에 설치하며, 실시간 Rerun 모드에서 자동으로 선택한다. 다른 환경에서
준비하려면 프로젝트 루트에서 실행한다.

```bash
bash scripts/run_dyna.sh -m pip install --target .deps-rerun --no-deps rerun-sdk==0.27.3
```

위 명령은 기존 dyna 환경의 numpy/pyarrow 등 의존성을 사용한다.

창을 클릭해 키 입력 포커스를 준 다음 아래 키로 조작한다. `Esc`로 종료한다.
프로그램은 YAML의 초기 ego pose에서 시작하며, 원래 ego 궤적 대신 직접 조작을 사용한다.
장면의 동적 물체는 시뮬레이터 시간에 따라 움직인다.

두 장면 사이를 전환하려면:

```bash
bash scripts/run_dyna.sh -m neural_scene.evaluation_realtime \
  --checkpoint outputs/training_run_v4_part1/best.pt \
  --scenarios outputs/realtime_scenarios/scene_0000.yaml \
              outputs/realtime_scenarios/scene_0010.yaml \
  --device cuda:0 --sensor lidar --speed 0.5 --angular-speed 30
```

GPU가 없으면 `--device cpu`를 사용한다. raycast만 CPU에서 실행하려면
`--device cuda:0 --sim-device cpu`를 지정한다. `--sim-device`를 생략하면
시뮬레이터도 모델과 같은 장치를 사용하며 YAML의 `run.device`보다 우선한다.

## 3. 조작 키

키를 누르는 동안 이동하며 놓으면 해당 이동을 멈춘다.

| 키 | 동작 |
|---|---|
| W / S | 로봇 로컬 X 전진 / 후진 |
| A / D | 로봇 로컬 Y 왼쪽 / 오른쪽 |
| H / G | 월드 Z 높이 증가 / 감소 |
| Y / T | 로봇 로컬 roll 증가 / 감소 |
| I / U | 로봇 로컬 pitch 증가 / 감소 |
| Q / E | 로봇 로컬 yaw 증가 / 감소 |
| N / B | 다음 / 이전 장면, 끝에서 순환 |
| Backspace | 현재 장면을 초기 상태로 재생성, prediction feedback 초기화 |
| 1 / 2 / 4 | prediction / 관측 / mesh 불투명도 조절 대상으로 선택 |
| + 또는 = / - | 선택한 대상의 불투명도 10% 증가 / 감소 |
| ↑ / ↓ | 선택한 대상의 불투명도 증가 / 감소 |
| 0 | 선택한 대상 숨김 / 이전 불투명도로 복원 |
| R | 카메라 중심을 현재 로봇 좌표의 원점으로 복귀 |
| Esc | 종료 |

`R`은 카메라 조작이며 로봇 pose나 feedback을 초기화하지 않는다.
카메라는 현재 로봇 좌표를 중심으로 표시한다. mesh도 같은 좌표계로 변환해 그린다.
이동이 지형·충돌 안전 검사에서 거부되면 로봇에 적용하지 않고 창 제목에
`motion blocked`를 표시한다. 조작 창 제목에는 frame 번호와 현재 추론 시간
`infer`(ms), 실행 시작부터의 평균 `avg`(ms), 누적 추론 시간 `total`(초),
전체 프레임 처리 시간 `loop`(ms)가 나온다. 추론 시간은 입력 전송·정렬·voxel화·
모델 실행·feedback 상태 갱신을 포함하며 raycast와 시각화는 제외한다.
CUDA는 측정 전후 동기화하며 첫 프레임도 포함한다. 평균과 누적값은 장면 전환이나
Backspace 이후에도 실행 전체를 기준으로 유지된다. `loop`는 raycast와 viewer 갱신을
포함하고 센서 주기를 맞추는 대기는 제외한다.

## 4. 주요 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--checkpoint` | 필수 | inference용 `best.pt` 또는 `last.pt`; `latest.pt`는 사용하지 않음 |
| `--scenarios` | 필수, 하나 이상 | DynaRaySim YAML 경로 목록 |
| `--device` | `cuda:0` | 모델 장치 |
| `--viewer` | `vispy` | `vispy` 또는 `rerun`; Rerun은 별도 키보드 조작 창 사용 |
| `--sim-device` | 모델 장치 | raycast/시뮬레이터 장치 |
| `--sensor` | YAML의 첫 센서 | 사용할 sensor ID |
| `--speed` | `0.5` | 이동 속도, m/s |
| `--angular-speed` | `30` | 회전 속도, deg/s |
| `--no-feedback` | 꺼짐 | 매 프레임 모델 상태를 초기화해 이전 예측 없이 추론 |
| `--voxels` | 꺼짐 | 점군 대신 grid의 점유 voxel 표시 |
| `--voxel-size` | `0.05` | voxel 한 변 길이, m |
| `--grid-shape` | `64 64 64` | XYZ 셀 수 |
| `--grid-origin` | `-1.6 -1.6 -1.6` | 로봇 좌표계에서 grid의 최소 모서리, m |
| `--prediction-opacity` | `0.9` | prediction 불투명도, 0–1 |
| `--observation-opacity` | `0.7` | 관측 불투명도, 0–1 |
| `--mesh-opacity` | `0.25` | mesh 불투명도, 0–1 |
| `--max-points` | `200000` | layer별 표시 점/voxel 개수 제한; 추론 입력은 유지 |
| `--point-size` | `3` | 표시 점 크기 |
| `--distance` | `6` | 초기 카메라 거리 |
| `--max-frames` | 제한 없음 | 처리할 총 프레임 수, 장면 전환 후에도 누적 |

grid는 checkpoint에서 자동으로 읽지 않는다. 다른 학습 설정의 모델을 사용하면
학습 데이터 `dataset.json`의 `grid`와 세 옵션을 맞춘다. 기본 grid의 범위는
각 축 `[-1.6, 1.6)` m다. 그 밖의 관측은 모델 voxelization에서 제외된다.

feedback에는 **직전 모델 예측**만 현재 pose로 정렬해 사용한다.
과거 원시 관측 전체를 쌓지 않는다. 장면 전환·초기화 시 feedback도 초기화된다.
처리 주기는 선택한 센서의 `frequency_hz`를 따른다. 예제는 10 Hz지만,
추론이 100 ms보다 오래 걸리면 실제 실행이 느려진다. 놓친 관측을 대기열에 쌓지 않는다.

초기 실행을 짧게 확인하려면 위 실행 명령에 `--max-frames 20`을 추가한다.
이 옵션도 GUI를 연다. 프로그램이 종료되면 창도 닫힌다.

## 5. 조이스틱

`dyna`에 pygame을 설치한 뒤 실행 명령에 옵션을 추가한다.

```bash
python -m pip install pygame

bash scripts/run_dyna.sh -m neural_scene.evaluation_realtime \
  --checkpoint outputs/training_run_v4_part1/best.pt \
  --scenarios outputs/realtime_scenarios/scene_0000.yaml \
  --device cuda:0 --joystick 0 \
  --joystick-axes 1 0 3 4 5 2 \
  --joystick-signs -1 -1 -1 1 1 -1 \
  --deadzone 0.15
```

축 번호·부호는 **로컬 X, 로컬 Y, 월드 height, 로컬 roll, 로컬 pitch, 로컬 yaw**
순서다. 위 값은 기본 매핑이며 장치마다 축 번호·방향을 맞춰야 한다.
`--joystick`은 0부터 시작하는 SDL 장치 번호다. 축 번호는 중복 지정할 수 있다.
부호는 각 축에 `1` 또는 `-1`, deadzone은 `0 이상 1 미만`이다.
키보드와 조이스틱 입력은 합쳐지고 각 축은 [-1, 1]로 제한된다.
연결 해제는 오류로 실행을 종료한다. 장면 전환·초기화는 키보드로 수행한다.

## 6. 실행 문제 확인

| 증상 | 확인할 사항 |
|---|---|
| `/home/intern/...` 파일을 못 찾음 | 1절의 YAML 사본 생성 후 `../outputs/realtime_scenarios` 경로로 실행 |
| `Unknown ego settings: ['terrain_alignment']` | 1절의 준비 명령으로 이전 옵션을 제거한 사본 생성 |
| `MinkowskiEngine` import 실패 | 프로젝트 루트에서 `bash scripts/run_dyna.sh ...` 사용, 의존성 폴더 확인 |
| `CUDA unavailable` | `bash scripts/run_dyna.sh scripts/check_gpu.py --require-cuda`로 확인하거나 `--device cpu` 선택 |
| GLFW/display 오류 | 데스크톱 세션에서 실행하고 `DISPLAY` 확인; 이 명령에는 headless 모드가 없음 |
| 키 입력이 반응하지 않음 | VisPy 창 클릭 후 입력; 이동 거부 시 창 제목의 `motion blocked` 확인 |
| 관측·예측이 비어 있음 | 초기 pose/센서 방향·range, grid 범위 확인; 새 장면으로 전환하거나 Backspace로 초기화 |
| 동작이 느림 | 창 제목의 처리 시간 확인; `--max-points`는 렌더링 부하만 줄이며 추론 속도를 보장하지 않음 |
| sensor/환경 설정 오류 | YAML sensor ID, `run.environments: 1`, 단일 ego 설정 확인 |
| joystick 축 부족/장치 없음 | SDL 장치 번호와 축 매핑, 연결 상태 확인 |

동적 물체의 명령 갱신 및 keyframe 시각을 거쳐 센서 시각까지 나누어 진행한다.
따라서 센서 프레임 사이의 명령 갱신 시각을 건너뛰지 않는다.

2026-11-11 기준 v4 part1 checkpoint와 scene_0000에서 CUDA 추론 및 Rerun 연결을
20프레임 실행해 정상 종료를 확인했다. 명령 갱신 시각 분할 회귀 테스트와 CPU
raycast 테스트를 포함해 관련 테스트 6개가 통과했다. 키보드 직접 조작 및
조이스틱 하드웨어 검증은 별도다.
