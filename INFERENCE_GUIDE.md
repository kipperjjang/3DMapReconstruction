# v3 모델 inference와 결과 재생

이 문서는 conda `dyna`, v3 데이터와 현재 프로젝트 코드를 기준으로 한다.
학습 폴더의 `best.pt`를 읽어 inference 결과를 **export**한 다음,
export 폴더를 DynaViewer로 **play**한다.

## 1. 환경과 checkpoint 선택

아래 명령은 모두 프로젝트 루트에서 실행한다.

```bash
cd "/home/intern/Neural Scene unet"
conda activate dyna

# 실제 사용할 학습 폴더를 지정한다.
RUN_DIR="outputs/training_run_v3_cp"
# 원래 학습 폴더를 사용한다면:
# RUN_DIR="outputs/training_run_v3"
```

`best.pt`는 저장 시점까지 validation F1이 가장 높았던 가중치다.
학습 도중에도 사용할 수 있으며, inference는 별도 모델을 로드하므로
진행 중인 학습 모델의 가중치나 BatchNorm 통계에 영향을 주지 않는다.
저장은 임시 파일 작성 후 교체하는 방식이다.

학습 중에는 `best.pt`가 갱신될 수 있다. 재현 가능한 비교와 export의 checkpoint
hash 기록을 위해 먼저 사본을 만든다. 아래 `cp`는 같은 이름의 이전 사본을 덮어쓴다.

```bash
mkdir -p outputs/inference_checkpoints
cp "$RUN_DIR/best.pt" outputs/inference_checkpoints/v3_best_preview.pt
```

`best_validation.json`에서 best가 선택된 update와 지표를 확인할 수 있다.
`latest.pt`는 optimizer·scheduler 등을 포함한 학습 재개용 파일이며,
현재 inference loader에는 `best.pt` 또는 `last.pt`를 사용한다.

## 2. 관측·예측·GT를 함께 export

처음에는 validation 장면 하나의 100프레임만 확인한다.
`--output`은 **아직 존재하지 않는 폴더**여야 한다. 재실행할 때 새 이름을 지정한다.

GPU에서 실행:

```bash
NEURAL_SCENE_DYNA_DEPS="$PWD/.deps-dyna-cuda" \
LD_LIBRARY_PATH="/home/intern/cuda-11.7/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
bash scripts/run_dyna.sh -m neural_scene.evaluation export \
  --checkpoint outputs/inference_checkpoints/v3_best_preview.pt \
  --dataset outputs/training_dataset_v3 \
  --output outputs/v3_preview \
  --split validation \
  --scene scene_0000 \
  --limit 100 \
  --device cuda:0
```

CPU에서 실행:

```bash
bash scripts/run_dyna.sh -m neural_scene.evaluation export \
  --checkpoint outputs/inference_checkpoints/v3_best_preview.pt \
  --dataset outputs/training_dataset_v3 \
  --output outputs/v3_preview_cpu \
  --split validation \
  --scene scene_0000 \
  --limit 100 \
  --device cpu
```

`run_dyna.sh`는 dyna 환경 및 workspace의 의존성을 설정한다. 단순히
`conda activate dyna`만 하는 것보다 이 wrapper를 사용하면 ME backend 경로가 명확하다.
GPU export는 학습과 GPU 메모리를 공유하며 학습 속도에도 영향을 줄 수 있다.
CPU export는 GPU를 사용하지 않지만 느릴 수 있다.

| 옵션 | 의미 |
|---|---|
| `--dataset` | `dataset.json`이 있는 데이터 폴더. v3 경로를 명시한다. 코드 기본값은 v2다. |
| `--split validation` | 학습 중 모델 확인용 validation split |
| `--scene scene_0000` | 해당 split 안의 장면 하나만 선택 |
| `--limit 100` | 선택된 **각 장면의 처음 100프레임**. 전체 합계 제한이 아니다. |
| `--no-feedback` | 이전 예측을 입력에 넣지 않는 비교 실험 |

전체 validation은 `--scene`과 `--limit`을 모두 생략한다.
v3의 validation은 4개 장면, 총 20,000프레임이므로 시간과 export 저장 공간이 더 필요하다.
최종 test 평가에는 `--split test`를 사용하고, validation의 `scene_0000` 옵션은 제거한다.
중간 설정 선택에는 validation을 사용하고 test는 최종 평가에 사용한다.
export 결과는 학습 데이터와 별도 저장되며, 10GB 수집 상한에 의해 자동 제한되지 않는다.

추론은 `.eval()`과 `torch.no_grad()`에서 수행한다. 각 scene의 첫 프레임은
현재 관측만 입력하고, 다음부터 이전 모델 예측을 DynaMapping 상대 변환으로 현재
robot frame에 맞춰 병합한다. 모든 시간 좌표는 `t=0`이며 scene마다 feedback을 초기화한다.

## 3. Export 결과 재생

위 GPU 예시가 정상 완료된 후 실행한다. CPU 예시를 사용했다면 입력 폴더를
`outputs/v3_preview_cpu`로 바꾼다.

```bash
python -m neural_scene.evaluation play --input outputs/v3_preview

# 예측값만 표시 (GT·관측 비교 레이어 숨김)
python -m neural_scene.evaluation play \
  --input outputs/v3_preview --foreground prediction --reference none

# 5cm occupied voxel box로 표시
python -m neural_scene.evaluation play --input outputs/v3_preview --voxels

# 예측과 현재 관측 비교
python -m neural_scene.evaluation play \
  --input outputs/v3_preview --reference observation

# 관측과 GT 비교
python -m neural_scene.evaluation play \
  --input outputs/v3_preview --foreground observation --reference target

# 반복 재생
python -m neural_scene.evaluation play --input outputs/v3_preview --fps 10 --loop
```

현재 환경에 DynaViewer의 `viewer` 패키지와 VisPy GUI 환경이 필요하다.
`No module named viewer`가 발생하면 DynaViewer가 설치된 환경에서 재생한다.
예를 들어 기존 viewer 환경이 유지되어 있다면 다음 경로를 사용할 수 있다.

```bash
/home/intern/DynaRaySim/.venv/bin/python -m neural_scene.evaluation play \
  --input outputs/v3_preview
```

재생은 모델이나 MinkowskiEngine을 로드하지 않으므로 inference와 다른 환경에서 가능하다.
원격 서버의 VisPy 창을 보려면 GUI 세션 또는 X11 forwarding이 필요하다.
기본 색은 청록=예측, 주황=GT다. 주황은 같은 시각의 비교 대상이며 과거 frame이 아니다.
좌표는 현재 robot frame의 metre 단위다. 마우스로 회전·확대하고 `R`로 중심 복원,
`Esc`로 종료한다. 기본은 마지막 프레임을 유지하며 `--no-hold`로 종료할 수 있다.
`--max-points`는 화면 표시량만 제한한다.

## 4. 저장되는 내용과 지표

| 경로 | 내용 |
|---|---|
| `evaluation.json` | 완료 상태, checkpoint·dataset 경로와 SHA256, grid, 모델 설정, feedback 여부, frame 목록·지표 |
| `report.json` | 모델 및 관측 baseline의 전체·장면별 평가 지표 |
| `scene_000/frame_00000.npz` 등 | 각 프레임의 관측·예측·GT와 자세 |

NPZ에는 다음 항목이 들어간다.

| 항목 | 의미 |
|---|---|
| `prediction_points`, `confidence` | 예측 voxel centroid와 같은 행에 대응하는 최종 점유 확률 |
| `current_points`, `target_points` | 현재 관측과 GT centroid, 현재 robot frame의 XYZ(m) |
| `robot_pose` | robot frame에서 world frame으로의 4×4 변환 |
| `transformation` | 이전 robot frame에서 현재 robot frame으로의 4×4 변환 |
| `timestamp_ns`, `frame_id` | 수집 시각과 scene 내부 frame 번호 |

평가는 같은 voxel을 점유하는지로 TP·FP·FN을 계산한다.
Precision은 `TP/(TP+FP)`, recall은 `TP/(TP+FN)`, F1은 `2TP/(2TP+FP+FN)`이다.
Centroid distance와 height MAE는 **예측과 GT가 같은 voxel에 있을 때만** 계산하며
단위는 m다. 전체 점군의 Chamfer distance는 아니다. 매칭이 없으면 해당 평균 오차는
`null`이다. 관측 baseline은 현재 관측만 voxel화한 결과다.
CPU/GPU 수치 차이와 짧은 구간의 편향을 고려해 최종 비교는 같은 backend·전체 구간으로 한다.

## 5. 자주 발생하는 오류

| 증상 | 원인과 처리 |
|---|---|
| `evaluation.json` 없음 | `play --input`에 학습 폴더를 넣은 경우. 먼저 export하고 export 폴더를 지정한다. |
| `FileExistsError` | export 출력 폴더가 이미 존재한다. 새 폴더 이름을 사용한다. |
| `Expected a completed ... export` | export가 중단됐거나 진행 중이다. `evaluation.json`의 `status=completed` 이후 재생한다. 중단된 export의 자동 재개는 지원하지 않는다. |
| Scene이 split에 없음 | `--scene`과 `--split` 조합을 확인한다. v3 validation 장면은 0000, 0010, 0020, 0030이다. |
| CUDA unavailable / OOM | GPU 접근·메모리를 확인하거나 CPU 명령을 사용한다. |

## 6. 별도 센서 데이터의 단일 프레임 inference

`neural_scene infer`는 GT 없이도 현재 관측 NPZ를 입력받는다.
입력 `current_points`는 현재 robot frame의 `[N,3]` XYZ(m)여야 한다.
아래 `my_frame.npz`는 사용자가 준비한 입력 파일을 뜻한다.

```bash
bash scripts/run_dyna.sh -m neural_scene infer \
  --checkpoint outputs/inference_checkpoints/v3_best_preview.pt \
  --input my_frame.npz \
  --output outputs/my_prediction.npz \
  --device cpu
```

다음 프레임부터 `--previous-prediction outputs/my_prediction.npz`를 추가하려면
입력 NPZ에 이전→현재 robot frame 변환 `transformation`(4×4)이 있어야 한다.
출력은 `points`, `confidence`, `origin`, `voxel_size_m`, `point_frame`, `input_policy`이며
기존 출력 파일은 덮어쓰지 않는다. 이 NPZ 하나는 `evaluation play`의 입력 폴더 형식과 다르다.

## 7. evaluation.py 전체 명령과 옵션

현재 [evaluation.py](neural_scene/evaluation.py)의 CLI 정의를 기준으로 한다.
상대 경로는 명령을 실행한 현재 디렉터리 기준이다. 이 문서의 예시는 프로젝트 루트에서 실행한다.

### 명령 선택과 도움말

| 명령·옵션 | 입력값 | 의미 |
|---|---|---|
| `export` | 아래 export 옵션 | checkpoint를 로드해 추론·평가하고 관측·예측·GT·지표를 저장한다. |
| `play` | 아래 play 옵션 | 완료된 export를 읽어 시각화한다. 모델 추론은 하지 않는다. |
| `-h`, `--help` | 값 없음 | 해당 명령의 도움말을 출력하고 종료한다. 최상위·export·play 모두 지원한다. |

```bash
python -m neural_scene.evaluation --help
python -m neural_scene.evaluation export --help
python -m neural_scene.evaluation play --help
```

`export` 또는 `play` 중 하나를 반드시 지정해야 한다.
표의 **값 없음** 옵션은 붙이면 켜지는 flag다. `--voxels true`처럼 값을 덧붙이지 않는다.

### export: 모든 옵션

기본 형태: `python -m neural_scene.evaluation export [옵션]`

| 옵션 | 입력값·허용 범위 | 필수 / 기본값 | 입력값의 의미와 동작 |
|---|---|---|---|
| `--checkpoint PATH` | checkpoint 파일 경로 | **필수** | 모델 가중치·구조·입력 정책을 읽는다. 현재 loader와 호환되는 `best.pt` 또는 `last.pt`를 지정한다. |
| `--dataset DIR` | `dataset.json`이 있는 디렉터리 | `outputs/training_dataset_v2` | 평가할 데이터셋. v3는 `outputs/training_dataset_v3`를 명시한다. |
| `--output DIR` | 아직 존재하지 않는 디렉터리 경로 | **필수** | `evaluation.json`, `report.json`, 프레임 NPZ를 저장할 새 폴더. 부모 폴더는 자동 생성한다. 기존 폴더에 이어 쓰지 않는다. |
| `--split NAME` | `train`, `validation`, `test` 중 하나 | `validation` | 데이터셋의 평가 대상 split. `train`은 학습 데이터 확인, `validation`은 중간 확인, `test`는 최종 평가에 사용한다. |
| `--scene NAME` | scene 이름 문자열, 예: `scene_0000` | 생략 시 선택 split의 모든 scene | 지정한 split에 실제로 존재하는 scene 하나만 평가한다. split이 다르거나 이름이 없으면 오류다. |
| `--device DEVICE` | PyTorch device 문자열. 일반적으로 `cpu`, `cuda:0` | `cpu` | 모델·Tensor 연산 장치. `cuda:0`은 프로세스에서 보이는 첫 GPU다. CUDA 접근과 해당 ME backend가 필요하다. |
| `--limit N` | 1 이상의 정수, 예: `100` | 생략 시 각 scene 전체 | 각 scene에서 처음 N프레임까지 평가한다. N이 scene 길이보다 크면 해당 scene 전체를 평가한다. 전체 합계나 시작 frame 번호가 아니다. |
| `--no-feedback` | 값 없음 | 꺼짐: 이전 예측 feedback 사용 | 지정하면 직전 모델 예측을 다음 입력에 넣지 않는다. 각 frame을 현재 관측만으로 추론하는 비교 실험이다. 결과 manifest의 `feedback`은 `false`가 된다. |
| `-h`, `--help` | 값 없음 | 생략 | export 도움말을 출력하고 종료한다. |

`export`는 입력 데이터의 grid와 checkpoint의 pruning 설정을 사용한다.
`--alpha`, `--origin`, `--previous-prediction`은 이 명령의 옵션이 아니다.
앞 절의 별도 `neural_scene infer` 명령과 구분한다.

### play: 모든 옵션

기본 형태: `python -m neural_scene.evaluation play [옵션]`

| 옵션 | 입력값·허용 범위 | 필수 / 기본값 | 입력값의 의미와 동작 |
|---|---|---|---|
| `--input DIR` | 완료된 export 디렉터리 경로 | **필수** | `evaluation.json`이 있고 `status=completed`인 폴더. `best.pt`나 학습 폴더 경로가 아니다. |
| `--scene NAME` | export에 포함된 scene 이름, 예: `scene_0000` | 생략 시 export의 모든 scene | 저장된 결과 중 scene 하나만 재생한다. `scene_000` 같은 내부 저장 폴더 이름이 아닌 manifest의 원래 scene 이름이다. |
| `--renderer NAME` | `vispy`, `rerun` 중 하나 | `vispy` | DynaViewer 렌더러 선택. VisPy는 GUI 비교용이며 Rerun은 GUI 또는 `.rrd` 기록에 사용한다. 선택한 렌더러의 의존성이 설치되어 있어야 한다. |
| `--foreground NAME` | `prediction`, `observation`, `target` 중 하나 | `prediction` | 청록 레이어에 표시할 점군. 각각 모델 예측, 현재 관측, GT를 뜻한다. |
| `--reference NAME` | `target`, `observation`, `none` 중 하나 | `target` | 주황 비교 레이어. 각각 GT, 현재 관측, 비교 레이어 숨김이다. **예측만 보려면 `--foreground prediction --reference none`**을 사용한다. |
| `--voxels` | 값 없음 | 꺼짐: point 표시 | 지정하면 선택된 점군을 dataset grid의 occupied voxel box로 표시한다. v3는 5cm다. 표시용 좌표는 voxel 중심이며 export된 예측값·지표는 바뀌지 않는다. |
| `--fps F` | 유한한 양수, 예: `10`, `2.5` | `10.0` | 화면 재생의 목표 초당 frame 수. 느리게 보려면 낮춘다. 렌더링·파일 로딩이 느리면 실제 fps는 낮아질 수 있다. 수집 timestamp나 inference 결과는 바꾸지 않는다. headless에서는 이 대기를 적용하지 않는다. |
| `--max-points N` | 1 이상의 정수 | `200000` | viewer에 전달할 표시 점 개수 제한. 파일의 포인트·추론 결과·평가 지표를 줄이지 않는다. |
| `--point-size S` | 유한한 양수 | `3.0` | viewer의 점 표시 크기. 수치가 클수록 점을 크게 표시하며 구체적인 표현은 renderer에 따른다. voxel 크기를 바꾸는 옵션이 아니다. |
| `--distance D` | 유한한 양수 | `6.0` | viewer의 초기 카메라 거리 설정. 화면의 초기 관찰 거리를 조절하며 데이터 좌표·센서 거리·crop 범위는 바꾸지 않는다. |
| `--loop` | 값 없음 | 꺼짐: 한 번 재생 | 선택한 모든 scene을 끝까지 재생한 후 처음부터 반복한다. headless와 함께 사용할 수 없다. |
| `--no-hold` | 값 없음 | 꺼짐: 마지막 화면 유지 | 한 번 재생을 마친 후 마지막 화면을 유지하지 않고 종료한다. 반복 중에는 끝나지 않으므로 `--loop`를 중단하는 옵션이 아니다. headless에서는 마지막 화면 유지 자체가 없다. |
| `--recording PATH` | 새 `.rrd` 파일 경로, 예: `outputs/v3_preview.rrd` | 생략 시 파일 기록 요청 없음 | Rerun 결과 저장 경로. `--renderer rerun`이 필요하다. 부모 폴더는 자동 생성하며 기존 파일은 덮어쓰지 않는다. |
| `--headless` | 값 없음 | 꺼짐: 화면 표시 | 창을 열지 않고 Rerun 기록을 만든다. `--renderer rerun`과 `--recording PATH`가 모두 필요하며 `--loop`와 함께 사용할 수 없다. |
| `-h`, `--help` | 값 없음 | 생략 | play 도움말을 출력하고 종료한다. |

### 옵션 조합 예시

이미 export한 결과에서 예측 voxel만 5fps로 반복 재생:

```bash
python -m neural_scene.evaluation play \
  --input outputs/v3_preview \
  --foreground prediction --reference none \
  --voxels --fps 5 --loop
```

GUI 없이 예측만 Rerun 파일로 저장한다. Rerun 의존성이 설치된 환경에서 실행하고,
`--recording`에는 기존에 없는 파일 이름을 지정한다.

```bash
python -m neural_scene.evaluation play \
  --input outputs/v3_preview \
  --foreground prediction --reference none \
  --renderer rerun \
  --headless --recording outputs/v3_preview_prediction.rrd
```

이 명령은 export 결과를 시각화용 기록으로 변환한다. 모델을 다시 추론하지 않는다.

관련 문서: [TensorBoard와 저장 파일](TENSORBOARD_GUIDE.md), [v3 데이터 구성](DATASET_V3_GUIDE.md).
