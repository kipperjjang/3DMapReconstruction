# 센서 포인트 클라우드 첫 학습

새 관측과 GT를 timestep별 NPZ로 저장. 프로젝트 루트의 `train.py`에서 데이터 경로와 학습 옵션 설정. 아래 명령은 프로젝트 루트에서 실행.

## 1. 학습에 필요한 모듈

Ubuntu/Linux 기준 설치 예시. 확인한 버전은 Python 3.11, NumPy 1.26.4, PyTorch 2.9.1, TensorBoard 2.21.0.

| 모듈 | 용도 |
|---|---|
| NumPy | NPZ 입출력, 포인트와 pose 전처리 |
| PyTorch | 모델 학습, 손실 계산, 체크포인트 |
| MinkowskiEngine | 희소 합성곱. 저장소의 `submodules/MinkowskiEngineCuda13` 소스를 빌드 |
| TensorBoard | 학습 로그 기록과 확인. `no_tensorboard=True`이면 생략 가능 |

Ubuntu에서 C++ 컴파일러, Python 헤더, OpenBLAS, Ninja 설치. Python 3.10–3.12의 `python3`로 가상환경 생성.

```bash
sudo apt install git build-essential python3-dev python3-venv libopenblas-dev ninja-build
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools==75.8.0
python -m pip install numpy==1.26.4 tensorboard==2.21.0
```

**CPU 학습** · CPU용 PyTorch 설치 후 같은 환경에서 MinkowskiEngine 빌드. `TrainConfig.device`는 `"cpu"`로 설정.

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python scripts/build_minkowski.py --output .deps
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
python -m neural_scene smoke --device cpu
```

**NVIDIA GPU 학습** · CUDA 13.0을 지원하는 NVIDIA 드라이버와 CUDA Toolkit 13.0(`nvcc` 포함) 설치. `CUDA_HOME`은 실제 Toolkit 경로로 지정. `TrainConfig.device`는 `"cuda:0"`으로 설정. [PyTorch 배포 버전](https://pytorch.org/get-started/previous-versions/) · [PyTorch 확장 빌드 도구](https://docs.pytorch.org/docs/stable/cpp_extension.html)

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
export CUDA_HOME=/usr/local/cuda-13.0
python scripts/build_minkowski.py --cuda --cuda-home "$CUDA_HOME" --output .deps-cuda
export PYTHONPATH="$PWD/.deps-cuda${PYTHONPATH:+:$PYTHONPATH}"
source scripts/cuda_env.sh
python scripts/check_gpu.py --require-cuda
python -m neural_scene smoke --device cuda:0
```

새 학습 터미널에서 가상환경 활성화 및 백엔드에 맞는 `PYTHONPATH` 설정. GPU 사용 시 `CUDA_HOME` 지정 후 `source scripts/cuda_env.sh` 실행.

## 2. 관측과 GT 저장

- 저장 경로는 `logs/<split>/<시퀀스 이름>/<timestep>.npz`.
- `train`, `validation`, `test`에 시퀀스를 각각 하나 이상 배치. 같은 장면의 연속 프레임은 같은 split에 배치. 시퀀스 이름은 split 전체에서 고유하게 지정.
- 파일은 `0.npz`, `1.npz`, `10.npz`처럼 숫자 순서로 로드.

| NPZ 키 | Shape | 내용 |
|---|---|---|
| `points_sensor` | `[N, 3]` | 센서 좌표계의 관측 XYZ(m). 프레임마다 점 개수 변경 가능. |
| `robot_pose` | `[7]` | 로봇 좌표에서 월드 좌표로의 pose. |
| `sensor_pose` | `[7]` | 해당 관측 시점의 센서 좌표에서 월드 좌표로의 pose. |
| `target_points_world` | `[M, 3]` | 월드 좌표계의 GT 표면 XYZ(m). 가려진 표면도 포함 가능. |
| `timestamp_ns` | 스칼라 정수 | 선택 필드. 시퀀스 내에서 증가하는 나노초 값. 생략 시 프레임 순번 사용. |

점 배열에는 유한한 실수 좌표 사용. 두 pose의 순서는 `[x, y, z, qx, qy, qz, qw]`, 위치 단위는 m. GT와 pose는 같은 월드 좌표계 사용.

```python
import numpy as np

# logs/train/route_a 디렉터리 생성 후 timestep별 저장
np.savez_compressed(
    "logs/train/route_a/0.npz",
    points_sensor=points_sensor,
    robot_pose=robot_pose,
    sensor_pose=sensor_pose,
    target_points_world=gt_surface,
)
```

학습 시 관측에 `inverse(T_world_robot) @ T_world_sensor` 적용. GT는 월드에서 현재 로봇 좌표계로 변환. 두 점군 모두 기본 5 cm voxel grid의 로봇 중심 범위 안에서 사용.

수집 완료 후 `logs/dataset.json` 생성.

```bash
python -m neural_scene.world_dataset logs
```

GT가 없는 정적 장면은 인덱싱 시 `--static-fusion-radius 2` 지정 가능. 같은 시퀀스에서 현재 관측과 앞뒤 최대 두 관측씩을 정렬해 의사 정답 생성. 이 경우 검증 점수도 의사 정답 기준. 프레임 추가 후 기존 `dataset.json`을 별도 위치로 이동하고 인덱싱 재실행.

## 3. 설정하고 학습 실행

프로젝트 루트의 `train.py`에서 `TrainConfig` 편집. [train.py](../train.py)

| 설정 | 지정할 값 |
|---|---|
| `dataset` | `dataset.json`이 있는 디렉터리. 예시는 `Path("logs")`. |
| `output` | 체크포인트와 TensorBoard 로그를 저장할 새 디렉터리. 기본값은 `Path("outputs/sensor_training")`. |
| `device` | GPU는 `"cuda:0"`, CPU는 `"cpu"`. |
| `updates`, `epochs` | 학습 길이. 하나만 지정. 둘 다 `None`이면 500 optimizer update 실행. |
| `batch_size`, `rollout` | update당 시퀀스 수와 시퀀스당 연속 프레임 수. `rollout`은 가장 짧은 train 시퀀스 길이 이하로 설정. |
| `eval_every` | validation을 실행할 update 간격. |
| `input_policy` | `None`이면 새 run은 `aligned_temporal_t01`, 체크포인트 사용 시 저장된 정책 적용. |
| `position_noise`, `no_mirror` | 관측 위치에 추가하는 노이즈 범위(m)와 XY 반전 증강 비활성화 여부. |
| `initial_lr`, `final_lr`, `seed` | 초기·최종 학습률과 난수 시드. |
| `no_tensorboard` | 기본값 `False`에서 TensorBoard 기록 활성화. |

- 첫 실행은 `resume=False`, `initial_checkpoint=None`으로 시작.
- 중단된 run은 같은 `output`과 설정으로 `resume=True` 지정.
- 기존 가중치로 새 run을 시작할 경우 새 `output`과 `initial_checkpoint` 지정.

```bash
python train.py
```

## 4. TensorBoard로 학습 확인

별도 터미널에서 프로젝트 루트로 이동 후 가상환경 활성화. `--logdir`에 `TrainConfig.output`과 같은 디렉터리 지정.

```bash
source .venv/bin/activate
python -m tensorboard.main \
  --logdir outputs/sensor_training --host 127.0.0.1 --port 6006
```

브라우저에서 `http://localhost:6006` 접속. Scalars에서 `train/loss`와 `validation/model/f1` 확인. 가로축 Step은 optimizer update. validation F1이 가장 높은 체크포인트는 `output/best.pt`에 저장.
