# 현재 구현 지도: 데이터 흐름·shape·좌표계·알고리즘

> 기준: 2026-11-11 작업 디렉터리의 실제 코드(미커밋 변경 포함). `neural_scene/`와 `scripts/`를 기준으로 작성했다. `project_log/**/source/`는 과거 실행의 코드 사본이므로 현재 실행 코드와 구분한다. 아래 수치는 별도 표시가 없으면 클래스/CLI 기본값이며, 실행 스크립트나 checkpoint 설정이 우선한다.

## 0. 이 문서의 표기와 읽는 순서

- **처음 읽기:** §1 전체 흐름 → §2 좌표계 → §4 전처리 → §5 모델 → §6 출력.
- **학습 이해:** §3 저장 데이터 → §7 loss → §8 학습 루프.
- **수정 요청 준비:** §9 실행 분기 → §11 수정 위치 지도 → §12 추적 예시.
- `#[N, 3]`은 배열의 shape. `#[]`는 scalar. `List[B]{#[N_b, 3]}`은 길이 B인 가변 길이 점군 목록이다.
- `N`: 관측 점 수, `P`: 직전 예측 점 수, `G`: GT 점 수, `V`: 입력 occupied voxel 수, `K`: 출력 voxel 수. 모두 프레임/배치에 따라 변한다.
- `B`: batch size, `L`: rollout 길이, `C_l`: level별 feature 채널 수.
- `M_l`: decoder pruning **전** 후보 수, `K_l`: pruning **후** 생존 수. Sparse row 수는 공간 해상도와 별개다.
- `t`: 실제 프레임 시각. `τ`: sparse coordinate의 입력 출처 표식(현재=0, 이전=1). 코드의 time 열은 여기서 `τ`로 쓴다.
- `T_A_B #[4, 4]`: B 좌표를 A 좌표로 변환. 점 배열은 행 벡터이므로 `points @ R.T + translation`을 사용한다.
- 코드 링크는 이 문서 기준 상대 경로다. 표의 함수 이름을 해당 파일에서 검색하면 구현 위치를 찾을 수 있다.

## 1. 프로젝트가 하는 일과 전체 분기

- **입력:** robot 좌표계의 부분 관측 점군 + 알려진 robot pose 또는 프레임 간 변환.
- **핵심:** 직전 복원 결과를 현재 robot 좌표계로 정렬하고, 현재 관측과 함께 sparse 4D U-Net에 넣는다.
- **출력:** 현재 robot 주변의 국소 3D 표면 점군. 살아남은 공간 voxel당 하나의 점과 confidence를 만든다.
- 모델의 상태는 **직전 예측 점군 하나**다. world 전체를 누적 저장하는 지도/pose 추정은 이 네트워크 경로에 구현되어 있지 않다.

```text
                         [오프라인 데이터 생성]
scene geometry + route + LiDAR 설정
 → DynaRaySim ray casting → sensor 관측 → robot 좌표 변환 → crop
 → world 표면 GT → 현재 robot 좌표 변환 → voxel centroid
 → DynaMapping의 이전→현재 변환 + pose/time 저장
 → dataset.json + NPZ 또는 packed HDF5
                       │
          ┌────────────┴───────────────────┐
          ▼                                ▼
   experiment.main()                 evaluation export
   랜덤 B개 연속 L-frame rollout       scene별 처음부터 순차 추론
   augmentation → predict()           evaluate() → predict()
   loss → backward → Adam             metrics + 프레임 NPZ 저장
   → checkpoint                            → play → DynaViewer
          │
          └──────────── 학습된 checkpoint ───────────────────┐
                                                          ▼
 [단일 NPZ infer]                [실시간 evaluation_realtime]
 current_points                  키 입력 → pose 갱신 → 새 ray casting
 + 이전 prediction NPZ           → PointCloudFrame → NeuralSceneMapper
 → 저장된 transformation 적용    → update() → 화면 표시
          │                                  │
          └────────── 공통 네트워크 경로 ──────┘
                       ▼
관측 #[N,3] + 정렬된 직전 예측 #[P,3]
 → voxelize → sparse C #[V,5], F #[V,3]
 → encoder → generative decoder + skip union + occupancy pruning
 → offset head → robot 점군 #[K,3] + confidence #[K]
 → detach → 다음 프레임의 이전 예측으로 사용
```

| 책임 | 주요 파일 / 클래스·함수 |
|---|---|
| 격자와 강체변환 | [voxel.py](../neural_scene/voxel.py): `VoxelGrid`, `voxelize`, `transform_points`; [geometry.py](../neural_scene/geometry.py): `pose_matrix`, `previous_to_current` |
| 입력 포장 | [inputs.py](../neural_scene/inputs.py): `ReconstructionInput`, `merged_input` |
| 신경망 | [model.py](../neural_scene/model.py): `UNetConfig`, `ReconstructionUNet4D`, `ReconstructionResult`; [blocks.py](../neural_scene/blocks.py) |
| 순차 상태 관리 | [mapping.py](../neural_scene/mapping.py): `PointCloudFrame`, `MappingStep`, `NeuralSceneMapper` |
| 학습·평가 | [experiment.py](../neural_scene/experiment.py): `SequenceDataset`, `predict`, `evaluate`, `main`; [loss.py](../neural_scene/loss.py); [training.py](../neural_scene/training.py) |
| 파일 추론·재생·실시간 | [__main__.py](../neural_scene/__main__.py), [evaluation.py](../neural_scene/evaluation.py), [evaluation_realtime.py](../neural_scene/evaluation_realtime.py) |

## 2. 좌표계: 어디에서 어디로 이동하는가

### 2.1 세 가지 좌표와 단위

| 표현 | shape / 타입 | 의미 |
|---|---|---|
| sensor point | `#[N,3]` float, m | 센서 원점/축 기준 ray hit |
| robot point | `#[N,3]` float, m | 현재 robot 기준. 모델 입력·출력·GT의 공통 좌표계 |
| world point | `#[N,3]` float, m | 고정 장면 좌표. 원본 GT surface와 mesh |
| `robot_pose` | `#[4,4]`, 주로 float64 | `T_world_robot` |
| pose 벡터 입력 | `#[7]` | `[x,y,z,qx,qy,qz,qw]`; `pose_matrix()`가 quaternion 정규화 후 `#[4,4]`로 변환 |
| grid cell | `#[V,3]` int32 | origin 기준 voxel 정수 인덱스. metre 좌표와 다름 |
| centroid offset | `#[V,3]` float | voxel 내부 분수 위치. voxel 단위, 대략 `[0,1)` |

```text
p_sensor #[N,3]
  ── T_robot_sensor = inverse(T_world_robot) @ T_world_sensor ──▶ p_robot #[N,3]

GT_world #[G_world,3]
  ── inverse(T_world_robot) ──▶ GT_robot #[G_local,3] ── crop/centroid ──▶ target_points

pred_previous_robot #[P,3]
  ── T_current_previous = inverse(T_world_current) @ T_world_previous ──▶ aligned_previous #[P,3]
```

- `previous_to_current()`의 실제 계산:
  - `R_rel #[3,3] = R_current.T @ R_previous`
  - `d_rel #[3] = R_current.T @ (d_previous - d_current)`
  - `aligned #[P,3] = previous #[P,3] @ R_rel.T + d_rel #[3]`
- `initial_pose`로 두 pose의 기준을 함께 바꾸어도 상대변환에서는 그 기준이 소거된다.
- 정렬 자체는 점 수를 바꾸지 않는다. **정렬 후 voxelize의 crop에서** grid 밖 점이 제거된다.
- pose 추정·보간은 하지 않는다. 입력 pose와 extrinsic이 맞아야 한다.
- `NeuralSceneMapper.frequency_hz`는 설정값으로 저장된다. 네트워크 입력에 `dt`나 속도를 추가하지 않으며, `dt_s`는 timestamp 차이를 기록하는 값이다.

### 2.2 기본 grid의 정확한 범위

- `VoxelGrid(voxel_size=0.05, shape=(64,64,64), origin=(-1.6,-1.6,-1.6))`
- 축별 범위: **`[-1.6, 1.6)` m**, cell index **`0..63`**.
- 하한 포함 / 상한 제외. `grid.shape`의 각 축은 모델에서 **16의 배수**여야 한다.
- 이 격자는 매 프레임 **현재 robot 좌표계**에 놓인다. 고정 world 격자가 아니다.

## 3. 데이터는 어디서 시작되고 어떻게 저장되는가

### 3.1 수집 코드의 역할 분기

| 경로 | 장면 생성 / 공통 처리 |
|---|---|
| 초기 NPZ 데이터 | [prepare_training_data.py](../scripts/prepare_training_data.py), [prepare_dataset_v2.py](../scripts/prepare_dataset_v2.py): archive 생성/전처리 → frame NPZ |
| v3 packed 수집 | [collect_dataset_v3.py](../scripts/collect_dataset_v3.py): `make_scene`, `shared_gt`, `collect_scene` |
| v4 box 장면 | [collect_dataset_v4.py](../scripts/collect_dataset_v4.py): `smooth_route`로 Catmull-Rom 경로를 조밀하게 만든 뒤 호 길이 기준 일정 거리 샘플링; 접선 yaw; box 배치 시 경로와 충돌 검사 |
| v4 solids 장면 | [collect_dataset_solids.py](../scripts/collect_dataset_solids.py): v4 경로 재사용, 27종 OBJ를 각 2회 배치 + random box. `--assets`로 v4 수집에서 선택 |
| 전용 solids 검증 장면 | [collect_validation_solids.py](../scripts/collect_validation_solids.py): 별도 경로/장면 구성, 공통 packed 수집 경로 사용 |
| 수집 감독·분할 | [run_dataset_v4.py](../scripts/run_dataset_v4.py), [run_solids_pipeline.py](../scripts/run_solids_pipeline.py), [split_dataset_v4.py](../scripts/split_dataset_v4.py), [finalize_solids_dataset.py](../scripts/finalize_solids_dataset.py) |

**packed 수집의 중심 `collect_scene()` → 내부 `consume(packet)`**

```text
시뮬레이터 packet의 scans + states
 → status == 1인 유효 sensor hit 선택: #[N_ray,3] → #[N_hit,3]
 → sensor→robot 변환: #[N_hit,3] → #[N_hit,3]
 → DynaMapping.update(): 실제 상대변환 #[4,4] 추출·검증
 → current: crop → #[N,3] (동일 voxel의 여러 관측점 유지)

shared_gt(): world surface #[S,3] → cKDTree
 → robot 위치 반경 2.78m 이웃 검색 → #[S_near,3]
 → world→robot 변환 → grid crop → #[S_local,3]
 → centroids(): voxel별 평균 → target #[G,3]

current / target → packed.encode() → #[N,6] / #[G,6] uint8
 → transpose → HDF5에 append
```

- v4 `shared_gt()`는 composed GT의 static scene boundary를 만들고 경로 주변 ROI만 보관한다. LiDAR 가시성으로 GT를 제한하지 않는다.
- 공통 수집기는 10Hz timestamp, pose 중복, LiDAR world Z 정렬, DynaMapping 상대변환 일치를 검사한다. current/target이 비면 수집을 중단한다.
- 공통 수집기의 기본 split은 `scene_id % 10`: 0→validation, 1→test, 나머지→train. 별도 분할/확정 스크립트가 manifest를 구성할 수 있으므로 **학습 시 최종 기준은 `dataset.json`**이다.
- DynaMapping은 데이터 생성에서 변환을 제공한다. 학습 feedback에는 DynaMapping의 raw history가 아니라 네트워크 예측을 사용한다.

### 3.2 저장 shape와 복호화

| 저장 항목 | shape | 의미 |
|---|---|---|
| `dataset.json` | JSON 구조 | format, input_policy, grid, sequences와 split, 프레임 목록, shard 경로 |
| HDF5 `current` / `target` | `#[6, ΣN_f]` / `#[6, ΣG_f]` uint8 | 모든 프레임 점을 이어 붙인 배열 |
| `current_offsets` / `target_offsets` | `#[F_scene+1]` int64 | 프레임 f는 `[offsets[f]:offsets[f+1]]` 구간 |
| `robot_pose`, `sensor_pose`, `transformation` | 각각 `#[F_scene,4,4]` float64 | world pose 또는 이전→현재 상대변환 |
| `timestamp_ns` | `#[F_scene]` int64 | 프레임 시각, ns |
| 구형 frame NPZ | dict | `current_points #[N,3]`, `target_points #[G,3]`, `robot_pose #[4,4]`, `transformation #[4,4]`, `timestamp_ns #[]` |

[`packed.py`](../neural_scene/packed.py)의 각 point record:

```text
p_robot #[N,3]
 → c = floor((p-origin)/0.05)                         #[N,3] int32
 → q = clip(floor((((p-origin)/0.05)-c)*256),0,255)  #[N,3] uint8
 → [cx,cy,cz,qx,qy,qz]                              #[N,6] uint8
 → 저장: spatial sort + transpose                  #[6,N]
 → 읽기: frame 구간 선택 + transpose               #[N,6]
 → decode: (c + (q+0.5)/256)*0.05 + origin          #[N,3] float32
```

- midpoint 복원 양자화 오차: 축별 약 `0.05/512 m = 0.0977 mm` 이하라는 설계값.
- 관측 중복점은 보존한다. target은 수집 단계에서 voxel당 centroid 하나로 줄인다.
- **packed 모듈은 5cm·64³·origin -1.6을 상수로 사용**한다. 임의 grid 지원으로 바꾸려면 `VoxelGrid`만 수정해서는 부족하다.
- `SequenceDataset.frame()`은 packed/NPZ 분기를 감추고 같은 frame dict를 반환한다. 최대 512개 frame을 LRU cache로 보관한다.
- loader는 scene 이름 중복 금지, 각 scene 12 frame 이상, train/validation/test 모두 존재를 요구한다.
- dataset의 input_policy는 지원되는 문자열인지 검사한다. 실제 `merged_input()`에는 **model.config.input_policy**를 전달하므로 저장 포맷의 policy와 실제 실행 policy를 혼동하지 않는다.

## 4. 현재 점군 → sparse 입력

### 4.1 `voxelize()`: 평균 위치를 feature로 바꾸기

```text
points #[N,3] float
 → normalized = (points-origin)/voxel_size         #[N,3]  voxel 단위
 → inside = all(0 <= normalized < grid.shape)      #[N] bool
 → normalized[inside]                             #[N_in,3]
 → cells = floor(normalized)                      #[N_in,3] int32
 → unique(cells, dim=0)
      coordinates                                 #[V,3] int32
      inverse                                     #[N_in] int64
      counts                                      #[V] int64
 → sums = index_add(normalized, inverse)            #[V,3]
 → offsets = sums/counts[:,None] - coordinates     #[V,3] float
```

- 같은 voxel에 들어간 점의 **centroid가 cell 안의 어디에 있는지** 3개 값으로 저장한다.
- `counts`는 voxelize 결과에 있지만 신경망 feature에는 넣지 않는다. 색상·intensity·normal도 현재 입력 feature에 없다.
- 예: `origin=-1.6`, size=0.05에서 점의 한 축 `-1.575, -1.565` → normalized `0.5,0.7` → cell `0` → offset `0.6` → centroid `-1.57 m`.

### 4.2 `merged_input()`의 두 정책

| policy | 연산 | 시간 열 |
|---|---|---|
| 기본 `aligned_temporal_t01` | 현재와 정렬된 이전 예측을 **각각 voxelize**, 이후 row concatenate | 현재 `τ=0`, 이전 `τ=1` |
| legacy `aligned_merge_t0` | `cat(current, previous)` 후 함께 voxelize | 모두 `τ=0` |

```text
measurements:     List[B]{#[N_b,3]}
aligned_previous: List[B]{#[P_b,3]} 또는 None

기본 정책, batch b:
 current  → C_current #[Vcur_b,3], offsets #[Vcur_b,3]
 previous → C_prev    #[Vprev_b,3], offsets #[Vprev_b,3]
 → C에 batch 열과 time 열 추가: [b,x,y,z,τ]
 → 모든 batch/time row 연결

ReconstructionInput
 coordinates C #[V_total,5] int32
 features    F #[V_total,3] float32
 batch_size B, grid VoxelGrid, input_policy str
 → .sparse() → ME.SparseTensor
```

- `Vcur_b`, `Vprev_b`는 각각 batch b의 현재/이전 voxel 수다.
- 동일 공간 cell이어도 현재/이전의 `τ`가 다르면 서로 다른 sparse row다.
- temporal 정책은 두 출처를 입력에서 평균하지 않는다. legacy 정책은 각 실제 점이 평균에 참여하므로 현재 원시점 수와 이전 예측점 수의 비중이 영향을 준다.
- `τ=0/1`은 초 단위 시간이 아니다. 둘 다 **현재 robot 좌표계**다.
- sparse 입력은 dense `#[B,3,64,64,64,2]`를 만들지 않는다. 점유 좌표 `C`와 그 row에 대응하는 feature `F`만 저장한다.

## 5. 신경망: sparse 4D generative U-Net

### 5.1 encoder와 decoder의 큰 구조

`ReconstructionUNet4D.forward(packed)`의 기본 채널은 `(32,64,128,256,256)`이다. 4D는 **x,y,z,τ**이며 batch 열은 convolution 차원에 포함되지 않는다.

```text
input C #[V,5], F #[V,3]
 E0 #[V0,32] ──────────────────────────────────────────────┐ skip0
  ↓ down: spatial stride 2                                 │
 E1 #[V1,64] ───────────────────────────────────────┐       │ skip1
  ↓                                                │       │
 E2 #[V2,128] ──────────────────────────────┐        │       │ skip2
  ↓                                        │        │       │
 E3 #[V3,256] ─────────────────────┐        │        │       │ skip3
  ↓                               │        │        │       │
 E4 #[V4,256] bottleneck           │        │        │       │
  ↑ generative up → union(skip3) → D3 → occupancy → prune   │
  ↑ generative up → union(skip2) → D2 → occupancy → prune   │
  ↑ generative up → union(skip1) → D1 → occupancy → prune   │
  ↑ generative up → union(skip0) → D0 → occupancy → prune ──┘
  → offset head → sigmoid → points
```

| 단계 | 공간 stride / 기본 논리 해상도 | block 입출력 feature shape | C shape |
|---|---|---|---|
| E0 | 1 / 64³ | `#[V0,3] → #[V0,32]` | `#[V0,5]` |
| E1 | 2 / 32³ | down으로 64채널 → block `#[V1,64]` | `#[V1,5]` |
| E2 | 4 / 16³ | down으로 128채널 → block `#[V2,128]` | `#[V2,5]` |
| E3 | 8 / 8³ | down으로 256채널 → block `#[V3,256]` | `#[V3,5]` |
| E4 | 16 / 4³ | down으로 256채널 → block `#[V4,256]` | `#[V4,5]` |
| D3 | 8 / 8³ | up 256 + skip 256 → `#[M3,512] → #[M3,256] → #[K3,256]` | 후보 `#[M3,5]` → 생존 `#[K3,5]` |
| D2 | 4 / 16³ | up 128 + skip 128 → `#[M2,256] → #[M2,128] → #[K2,128]` | `#[M2,5] → #[K2,5]` |
| D1 | 2 / 32³ | up 64 + skip 64 → `#[M1,128] → #[M1,64] → #[K1,64]` | `#[M1,5] → #[K1,5]` |
| D0 | 1 / 64³ | up 32 + skip 32 → `#[M0,64] → #[M0,32] → #[K0,32]` | `#[M0,5] → #[K0,5]` |
| offset | 공간1 / temporal2 | `#[K0,32] → #[K,3]` | `#[K,5]`, `τ=0` |

- 해상도는 공간 stride를 설명하는 **논리적 표현**이다. dense tensor 크기나 실제 활성 row 수를 뜻하지 않는다.
- Minkowski 좌표는 기본 격자의 정수 좌표이고 coarse level은 stride의 배수로 표현된다. 예: stride 4에서 `x=12`를 `x=3`으로 다시 쓰지 않는다.
- 일반 conv와 공간 down/up 동안 temporal stride는 1이다.
- 코드 `decoders[0..3]`, `stages[0..3]`는 이 표의 **D3 → D2 → D1 → D0** 순서다.

### 5.2 핵심 연산 블록

| 연산 | kernel / stride | 무엇이 바뀌는가 |
|---|---|---|
| `ConvBlock(in,out)` | 일반 4D conv `[3,3,3,3]`, stride 1을 두 번 | `F #[V,in] → #[V,out]`; 각 conv 뒤 BN→ReLU. 현재/이전 시간 이웃도 섞음 |
| `SpatialDownsample` | `[2,2,2,1]` / `[2,2,2,1]` | 공간 해상도 감소, 채널 변경, row 수 변화. 시간 stride 유지 |
| `SpatialUpsample` | generative transpose `[2,2,2,1]` / `[2,2,2,1]` | 공간 해상도 증가, **새 좌표 후보 생성**, 채널 변경 |
| occupancy head | kernel 1 / stride 1 | `#[M_l,C_l] → logits #[M_l,1]` |
| pruning | `sigmoid(logits[:,0]) >= α` | mask `#[M_l]` → 좌표/feature 중 `K_l` row 유지 |
| temporal offset head | `[1,1,1,2]` / `[1,1,1,2]` | 시간 0/1을 공간 voxel 하나로 축약, 3채널 출력 |
| legacy offset head | kernel 1 / stride 1 | 시간 축약 없이 3채널 출력 |

**sparse convolution 개념**

```text
출력 좌표 q에서:
 feature_out(q) = Σ_delta feature_in(q + delta) @ W_delta
                      ↑ 실제 존재하는 입력 좌표의 기여만 사용
```

- 일반 conv는 존재하는 support에서 feature를 계산한다. 미관측 공간을 채우는 후보는 **generative upsample과 skip union**에서 생긴다.
- decoder는 후보를 무조건 출력하지 않고 매 단계 occupancy로 솎는다.
- `SparseBatchNorm`: training 중 활성 row가 하나뿐이면 해당 연산에서 running statistics를 사용한다. 그 외에는 일반 BN. singleton 처리를 위해 점을 추가하지 않는다.

### 5.3 skip은 좌표 합집합 위의 concatenate

`union_concat(decoder, encoder)`:

```text
decoder: C_d #[Nd,5], F_d #[Nd,C]
encoder: C_e #[Ne,5], F_e #[Ne,C]

decoder feature → [F_d, 0] #[Nd,2C]
encoder feature → [0, F_e] #[Ne,2C]
 → MinkowskiUnion으로 합집합 좌표에서 합산
 → C_union #[M,5], F_union #[M,2C]
```

- 양쪽 모두 존재: `[decoder_feature, encoder_feature]`.
- 한쪽에만 존재: 없는 쪽 채널을 0으로 채움.
- row 번호를 맞춰 붙이지 않는다. **좌표**를 기준으로 결합한다.
- generative decoder와 encoder의 coordinate map이 다를 수 있어 단순 `ME.cat`을 쓰지 않는다.

## 6. pruning 결과 → 위치와 confidence → 다음 feedback

### 6.1 출력 위치 복원

```text
D0 생존 feature #[K0,32], C #[K0,5]
 → offset_head: 시간 0/1 축약 → raw_offsets #[K,3]
 → sigmoid → offsets.F #[K,3] (voxel 내부 위치)
 → offsets.C #[K,5] = [batch,x,y,z,0]
 → positions = (C[:,1:4] + F)*voxel_size + origin  #[K,3] m
 → batch 열로 분리 → List[B]{#[K_b,3]}
```

- temporal 정책의 최종 tensor stride는 `[1,1,1,2]`; legacy는 `[1,1,1,1]`이다. **공간 stride가 1**이라는 뜻이지 temporal stride도 1이라는 뜻은 아니다.
- 두 시간 row가 같은 공간 voxel에 있으면 축약 후 하나의 공간 출력이 된다. 그러므로 `K <= K0`일 수 있다.
- offset은 voxel 중심에서의 ±변위가 아니다. **cell의 lower corner에서부터의 0~1 분수 위치**다.

### 6.2 confidence의 정확한 정의

- 최종 D0 pruning **전** occupancy logit → sigmoid `#[M0]`.
- batch+x+y+z 키로 시간 row를 묶고 **최댓값(amax)** → 공간별 confidence.
- output 좌표와 키로 다시 매칭 → `confidence #[K]`.
- feature/coordinate row가 pruning 후 재정렬될 수 있어 row 순서만으로 대응시키지 않는다.
- confidence는 offset head 출력이나 GT와의 거리에서 나온 값이 아니다.

### 6.3 반환 구조와 빈 출력

| 구조 | 필드 |
|---|---|
| `OccupancyStage` | `logits`: pruning 전 SparseTensor(`C #[M_l,5]`, `F #[M_l,1]`), `keep #[M_l]` bool |
| `ReconstructionResult` | `offsets`(`C #[K,5]`, `F #[K,3]`), `confidence #[K]`, `stages` 최대 4개, `batch_size`, `grid`; `.points()`로 metre 점군 반환 |
| `MappingStep` | `reconstruction`, `points #[K,3]`, `aligned_previous #[P,3]`, `transformation #[4,4]`, `dt_s #[] 또는 None`, `frame_id` |

- 입력 전체가 비면 encoder를 실행하지 않고 빈 결과를 반환한다.
- 중간 decoder 단계에서 모든 후보가 제거되면 그 단계까지의 `stages`와 빈 결과를 반환한다.
- `EmptyOffsets`: `F #[0,3]`, `C #[0,5]`. CUDA에서 빈 coordinate map 등록 문제를 피하는 반환 객체다.
- 최소 생존 voxel 보장, GT voxel 강제 보존, 빈 예측을 관측으로 대체하는 fallback은 없다.
- feedback은 `result.points()`를 **detach**해서 저장한다. 다음 프레임에서 재정렬·재voxelize한다.

## 7. 정답과 loss: 무엇을 학습하는가

### 7.1 GT 포장과 좌표 키

```text
targets List[B]{#[G_b,3]} 현재 robot frame, m
 → target_voxels(): batch별 voxelize
 → GT coordinates #[Gv,4] = [batch,x,y,z]
 → GT offsets     #[Gv,3]

spatial_keys(C): #[N,4 또는 5] → #[N] int64
 key = ((batch*Sx + x)*Sy + y)*Sz + z
```

- 시간 열은 키에 넣지 않는다. 두 시간 입력 모두 **현재 장면의 표면**을 복원하도록 감독한다.
- `match_keys(query, reference)`는 sort→searchsorted로 membership mask `#[N_query]`와 reference row indices `#[N_query]`를 반환한다. indices는 membership이 참인 곳에서 사용한다.

### 7.2 Multi-scale occupancy loss

각 decoder stage의 spatial stride `s=[sx,sy,sz]`에 대해:

```text
GT fine cells #[Gv,3]
 → coarse GT cells = floor(GT_cells / s) * s       #[Gv,3]
 → 후보의 [batch,x,y,z]가 coarse GT에 존재?         labels #[M_l] bool
 → BCEWithLogits(logits[:,0], labels)              #[] stage loss
 → 실행된 stage loss들의 평균                     #[] L_occ
```

- BCE는 **pruning 전 생성된 후보**를 대상으로 계산한다.
- 후보 자체가 생성되지 않은 GT 좌표를 추가하는 연산은 없다.
- coarse stage의 두 시간 row가 같은 공간 좌표면 같은 GT label을 받는다.
- `α` 기본값은 0.5이며 학습 때도 예측 확률로만 pruning한다. hard mask 선택 자체를 미분하는 방식은 아니다.

### 7.3 Sub-voxel offset loss

```text
최종 output C #[K,5] ↔ GT C #[Gv,4] 공간 키 매칭
 → 공통 voxel J개 선택
 → e = predicted_offsets - gt_offsets             #[J,3]
 → norm(e, dim=1)                                 #[J]
 → mean                                          #[] L_offset

L_total = occupancy_weight * L_occ + offset_weight * L_offset
```

- 기본 가중치는 `1:1`이다. offset은 **mean Euclidean distance**, voxel 단위다(제곱오차가 아님).
- metre로 해석하려면 `voxel_size`를 곱한다.
- 공통 voxel이 없으면 offset loss는 0이다. occupancy loss는 생성된 stage가 있으면 남는다.
- 이 loss에는 별도 Chamfer distance, mesh loss, 명시적 시간 일관성 loss가 없다.

## 8. 학습: B개 episode × L개 연속 frame

중심 진입점은 `python -m neural_scene.experiment`의 `main()`이다.

```text
dataset.json → SequenceDataset → train scene별 observable_starts()
 → 모델/optimizer/scheduler 준비
 → 초기 validation

optimizer update 한 번:
  zero_grad()
  → scene B개 랜덤 선택(중복 가능), 각각 시작 frame 랜덤 선택
  → batch member별 mirror 부호를 한 번 선택, previous=[None]*B
  → age=0..L-1:
      연속 frame B개 읽기
      → augmented_frame()
      → predict(): 이전 예측 정렬 → merged_input() → model()
      → reconstruction_loss()
      → (loss/L).backward()             # 매 frame gradient 누적
      → previous = result.points().detach()
  → gradient finite 확인
  → optimizer.step()                    # L번이 아니라 1번
  → scheduler.step()
  → 주기적 validation → F1 최대이면 best.pt
  → latest.pt에 optimizer/RNG까지 저장

전체 학습 종료 → last.pt 저장 → best.pt 재로드 → test 평가 → report.json
```

- 배치는 `#[B,L,N,3]`으로 고정 padding하지 않는다. 각 age에서 `List[B]{#[N_b,3]}`를 포장한다.
- 첫 프레임에는 관측이 있는 rollout만 시작한다. 이후 blind frame은 포함하고 feedback으로 처리한다.
- recurrence를 detach하므로 시간 전체로 BPTT하지 않는다. 매 frame `loss/L`로 backward하여 L개 계산 그래프를 동시에 보관하지 않는다.
- loss는 stage/후보/일치점 평균이므로 batch member별 점 수가 다를 때 단순한 “scene별 loss의 균등 평균”과 같다고 가정하면 안 된다.

### 8.1 Augmentation과 optimizer

| 항목 | 실제 연산 / shape |
|---|---|
| mirror | rollout마다 x/y 부호 `mx,my∈{-1,+1}` 고정; `M=diag(mx,my,1,1) #[4,4]` |
| 점 반사 | current `#[N,3] @ M[:3,:3]`, target `#[G,3] @ M[:3,:3]` |
| 변환 반사 | 상대변환과 robot_pose를 `M @ T @ M`로 conjugation; `#[4,4]` 유지 |
| 위치 노이즈 | current에만 `uniform(-a,+a) #[N,3]` 추가. target에는 추가하지 않음 |
| optimizer | Adam, 초기 LR 0.01 → 최종 0.0001 |
| decay | `gamma=(final_lr/initial_lr)^(1/total_updates)`로 update마다 exponential decay |

- validation/test는 augmentation 없이 scene 처음부터 실행하고 scene마다 feedback을 초기화한다.
- `--epochs E`는 `ceil(train_frames/(B*L))*E` optimizer updates를 뜻한다. 각 프레임을 정확히 한 번씩 순회하는 epoch가 아니라 랜덤 rollout의 **유효 frame 통과량**이다.

### 8.2 설정 분기와 산출물

| 설정 | 결과 |
|---|---|
| 새 run | 기본 temporal 모델, 새 optimizer/RNG |
| `--initial-checkpoint` | 가중치 warm start, optimizer/run은 새로 시작 |
| `--resume` | `latest.pt`에서 가중치·optimizer·scheduler·RNG·step 복구; 설정/dataset 일치 검사 |
| `--evaluate-only` | initial checkpoint를 고정 평가; optimizer update 없음 |
| `--calibrate-bn EPISODES` | evaluate-only와 함께 사용. 깨끗한 train rollout으로 BN 통계 재추정; parameter gradient update 없음 |
| `--skip-test` | validation까지 실행, 최종 test 보류 |

| 파일 | 용도 |
|---|---|
| `best.pt`, `last.pt` | inference checkpoint: format, input_policy, config, model weights |
| `latest.pt` | resume checkpoint: 위 정보 + optimizer/scheduler/RNG/step/metadata |
| `configuration.json`, `source/`, `source_hashes.json` | 설정 및 실행 소스 snapshot |
| `training.jsonl`, TensorBoard events | update별 loss/LR/출력 수/평가 지표 |
| `best_validation.json`, `report.json` | best 선택 근거와 최종 test. `status=completed`로 완료 판단 |

- `load_checkpoint()`는 inference format만 허용한다. `latest.pt`를 그대로 inference 입력에 넣지 않는다.
- **기본값 층위:** `experiment` CLI는 B=2, L=12, 500 updates, noise=0.05m, eval=50이다. 현재 `scripts/train_v4.sh`는 B=4, L=12, 20,000 updates, noise=0.005m, eval=1,000으로 실행한다. 사용자 인자에 따라 달라질 수 있다.
- 실행 환경은 `scripts/run.sh`(기본 sp_conv), `scripts/run_dyna.sh`(기본 dyna)가 dependency 경로/CONDA/CUDA 라이브러리 경로를 준비한다. neural_scene의 연산 자체와 실행 환경 설정은 별도 책임이다.

## 9. 추론·평가·시각화의 기능 분기

### 9.1 파일 한 장 추론: `python -m neural_scene infer`

```text
inference checkpoint → model.eval()
input NPZ.current_points #[N,3]
 + 선택: 이전 output NPZ.points #[P,3]
 → 현재 input NPZ.transformation #[4,4]로 이전 예측 정렬
 → merged_input → model → result.points()[0] #[K,3]
 → output NPZ:
    points #[K,3], confidence #[K], origin #[3], voxel_size_m #[],
    point_frame='current_robot', input_policy
```

- `current_points`는 이미 robot 좌표여야 한다. raw `previous_points`를 자동 feedback으로 사용하지 않는다.
- `--alpha`로 pruning threshold, `--origin`으로 grid origin을 바꿀 수 있다. 이 CLI의 size/shape는 `VoxelGrid` 기본값을 쓴다.
- 기존 output 파일을 덮어쓰지 않는다.

### 9.2 메모리 내 순차 API: `NeuralSceneMapper.update(frame)`

```text
PointCloudFrame(points #[N,3], robot_pose #[4,4] 또는 #[7], timestamp_ns #[], frame_id)
 → pose·strict timestamp 증가 검사
 → 첫 frame: aligned_previous #[0,3], transformation=I, dt_s=None
 → 이후: previous_pose와 current_pose로 변환 계산 → previous_prediction 정렬
 → merged_input → model → MappingStep
 → state 갱신(previous_pose, timestamp, detached prediction)
```

- model.train()/eval() 선택은 호출자 책임이다. `track_grad=False`는 no_grad, True는 현재 결과의 gradient를 유지한다.
- `reset()`은 새 episode/scene에서 호출한다. `rollout_loss()`는 기본 12-step loss 평균만 반환하고 optimizer step은 하지 않는다.
- 학습/오프라인 평가는 mapper 대신 `experiment.predict()`를 쓰며, pose를 다시 계산하지 않고 저장된 `frame['transformation']`을 사용한다.

### 9.3 평가 export: `python -m neural_scene.evaluation export`

```text
SequenceDataset + checkpoint
 → evaluate(): scene마다 previous=[None]
 → frame load → predict → metric → save_frame callback
 → prediction/observation/GT NPZ + evaluation.json + report.json + timing.json
```

**평가 정의** (`counts_and_errors` → `summarize`):

- output voxel과 GT voxel의 공간 키 일치 수 → TP; 나머지 output → FP; 나머지 GT → FN.
- `precision=TP/(TP+FP)`, `recall=TP/(TP+FN)`, `F1=2TP/(2TP+FP+FN)`; 분모 0 보호 적용.
- 거리: 동일 voxel의 `offset 차이 * voxel_size` → Euclidean norm 평균(m).
- height MAE: 동일 voxel의 z 차이 절댓값 평균(m). **robot z 기준**이며 일반 pose에서는 world 수직축과 다를 수 있다.
- TP가 없으면 거리/height 평균은 `None`. 누락된 표면과 false positive는 그 거리 평균에 포함되지 않는다.
- 전체 지표는 TP/FP/FN과 오차 합을 누적한 뒤 계산한다. frame별 F1의 단순 평균이 아니다.
- measurement baseline은 현재 관측만 voxelize한 결과와 GT를 비교한다.
- `--no-feedback`: 이전 예측을 넣지 않음. `--limit`: 각 scene의 앞 N frame. scene/split 선택 가능.

**export frame NPZ**

| 필드 | shape |
|---|---|
| `prediction_points`, `confidence` | `#[K,3]`, `#[K]` |
| `current_points`, `target_points` | `#[N,3]`, `#[G,3]` |
| `robot_pose`, `transformation` | `#[4,4]`, `#[4,4]` |
| `timestamp_ns`, `frame_id` | 각각 `#[]` |

- `evaluation.json`: checkpoint/dataset SHA256, grid/policy, scene/frame 목록과 지표. running→completed로 확정한다.
- `InferenceTiming`: CUDA 작업 전후 synchronize, wall-clock 기록. export는 `predict()` 범위(전송·전처리·모델)를 측정하고 파일 읽기/metric/저장을 제외한다. warmup frame을 제외하지 않는다.

### 9.4 저장 결과 재생: `python -m neural_scene.evaluation play`

```text
completed evaluation.json → 프레임 NPZ
 → comparison_snapshot(): 의미별 layer 구성
 → ComparisonViewer → VisPy 또는 Rerun
```

- 이 경로는 checkpoint/모델 추론 없이 동작한다.
- 예측=청록, 관측=초록, target GT=주황. reference가 viewer의 history 슬롯을 써도 **동일 시각 비교 레이어**다.
- `--voxels`: point 위치를 cell로 변환 → unique → `(cell+0.5)*size+origin`으로 **box 중심** 생성. 학습 offset 위치와 표시용 voxel 중심은 다르다.
- mesh는 world vertices `#[Nv,3]`와 triangle indices `#[Nf,3]`. `mesh_in_robot()`에서 현재 robot 좌표로 변환해 점군에 맞춘다.
- layer/opacity/max-points는 표시 설정이다. 저장된 예측이나 평가 수치를 수정하지 않는다.

### 9.5 실시간: `python -m neural_scene.evaluation_realtime`

```text
scenario + checkpoint → LiveScene + NeuralSceneMapper + ComparisonViewer
 → Controls.command()
 → LiveScene.acquire():
     동적 object advance → integrate_pose(이동/회전) → scene._safe로 허용 여부 결정
     → WarpBackend.cast()
     → origin + range*direction                   #[N_ray,3] sensor
     → valid hit 선택 + extrinsic                 #[N,3] robot
     → points, pose, timestamp, frame_id
 → mapper.update() → 예측 #[K,3]
 → object mesh를 world로 합성 → viewer에서 robot 좌표로 변환
 → observation/prediction/mesh 갱신 → 다음 sensor 주기
```

- `LiveScene`은 한 environment와 선택된 한 sensor를 사용한다. 이동을 직접 제어하도록 scripted ego motion을 stationary로 전환한다.
- scene 변경/reset 시 live scene과 mapper를 새로 만든다. `--no-feedback`은 매 frame mapper를 reset한다.
- 이 경로의 inference timing은 `mapper.update()` 범위다. sensor casting/mesh 구성/렌더링을 포함한 전체 loop 시간과 구분한다.
- 실시간 표시 경로는 target surface를 생성해 F1을 평가하는 export 경로와 다르다.

## 10. 전체 shape 계약 빠르게 찾기

| 경계 | 입력 | 출력 | 좌표/단위 변화 |
|---|---|---|---|
| `pose_matrix` | `#[7]` 또는 `#[4,4]` | `#[4,4]` | XYZ+XYZW → 강체행렬 |
| `previous_to_current` | current/previous pose | `#[4,4]` | world pose → 이전 robot에서 현재 robot으로 |
| `transform_points` | `#[P,3]`, `#[4,4]` | `#[P,3]` | 이전 robot m → 현재 robot m |
| `voxelize` | `#[N,3]` | C `#[V,3]`, offset `#[V,3]`, counts `#[V]` | robot m → 정수 cell + 분수 위치 |
| `merged_input` | `List[B]{#[N_b,3]}`와 이전 예측 | C `#[V,5]`, F `#[V,3]` | batch/time 열 추가 |
| U-Net | C `#[V,5]`, F `#[V,3]` | C `#[K,5]`, F `#[K,3]`, conf `#[K]` | 후보 생성·pruning·시간 축약 |
| `.points()` | 최종 C/F | `List[B]{#[K_b,3]}` | cell+offset → 현재 robot m |
| `target_voxels` | `List[B]{#[G_b,3]}` | C `#[Gv,4]`, F `#[Gv,3]` | batch 포함, time 없음 |
| `reconstruction_loss` | result + targets | total/occupancy/offset `#[]` | occupancy와 voxel 단위 위치오차 |
| `packed.encode/decode` | `#[N,3]` / `#[N,6]` | `#[N,6]` / `#[N,3]` | m ↔ uint8 cell+양자화 offset |

## 11. 원하는 동작을 바꾸려면 어디를 고칠까

| 바꾸고 싶은 것 | 1차 수정 위치 | 함께 점검할 계약 |
|---|---|---|
| robot 주변 범위/voxel 크기 | `voxel.VoxelGrid`, dataset grid | packed의 ORIGIN/SIZE/64 상수, 수집 crop·GT·검증, 추론 CLI, shape 16배수 |
| 센서 extrinsic/pose 정렬 | 수집의 sensor→robot, `geometry.previous_to_current`, `voxel.transform_points` | T_A_B 방향, XYZW 순서, 단위, robot/world 구분 |
| 현재·이전 입력 합치는 방식 | `inputs.merged_input`, policy 상수 | `UNetConfig`, offset head의 temporal kernel/stride, checkpoint policy |
| 과거를 여러 frame 보관 | `NeuralSceneMapper` state, `experiment.predict` 및 학습 previous | time coordinate 허용 범위, input packing, temporal head, state reset, 새 checkpoint 구조 |
| intensity/normal/count 추가 | `ReconstructionInput.features`, `merged_input` | encoder0 입력 3채널 가정, 데이터 저장 포맷, augmentation, 새 학습 |
| 모델 깊이/폭/kernel | `UNetConfig`, `ReconstructionUNet4D.__init__`, `blocks.py` | 5개 level 검증, down/up 수와 skip, grid 배수, checkpoint 호환 |
| 빈 출력/과도한 생성 조절 | `UNetConfig.pruning_threshold`, `forward`의 keep | CLI infer는 `--alpha` 지원; occupancy supervision·생성 후보·confidence 정의 |
| skip 결합 | `blocks.union_concat` | coordinate union, 없는 feature 0 처리, tensor_stride 일치 |
| GT/위치오차 정의 | `loss.target_voxels`, `occupancy_labels`, `reconstruction_loss` | temporal label 무시 여부, voxel/m 단위, metric와 차이 |
| 노이즈·mirror·rollout 샘플링 | `experiment.augmented_frame`, `observable_starts`, `main` | 관측만 noise, rollout 내 mirror 고정, 상대변환 conjugation |
| 시간 gradient 연결 | 학습 loop의 `.detach()`, mapper의 state 저장 | 그래프 수명·메모리·backward 방식 전체 재설계 필요 |
| optimizer/LR/학습량 | `training.adam_with_decay`, `experiment.main`, `scripts/train_*.sh` | 스크립트 override, updates와 effective epochs, resume metadata |
| 지표나 best 선택 기준 | `counts_and_errors`, `summarize`, main의 best_f1 | 동일 voxel 거리의 한계, frame 평균과 전체 집계 차이 |
| 저장/export 필드 | `evaluation.export_evaluation`의 `save_frame`, `__main__.infer` | 파일 추론 output은 `points`, 평가 output은 `prediction_points` |
| 색상·표시·opacity | `evaluation.comparison_snapshot`, `evaluation_viewer.ComparisonViewer` | 표시용 voxel 중심과 실제 centroid 구분 |
| 실시간 이동·센서 입력 | `evaluation_realtime.integrate_pose`, `LiveScene.acquire` | simulator safety, sensor frequency, extrinsic, mapper reset |
| 장면/물체/경로/GT | `scripts/collect_dataset_v4.py`, `collect_dataset_solids.py`, 공통 `collect_scene` | geometry와 route 충돌, static boundary GT, split과 provenance |

**수정 요청 예시**

- “`inputs.merged_input()`에서 temporal 두 slice를 유지하되 count를 4번째 feature로 넣고, encoder0 입력 채널과 데이터 계약을 같이 바꿔줘.”
- “D0 pruning의 α만 바꿔 성능을 비교하고, export의 전체 F1과 출력 voxel 수를 보고 싶어.” 현재 α는 네 decoder 단계에 공통 적용되므로 **D0만** 조절하려면 stage별 설정을 추가해야 한다.
- “`counts_and_errors()`의 동일 voxel 위치오차 외에 전체 점군 거리를 추가하고, best 선택은 기존 F1을 유지해줘.”

## 12. 한 프레임을 끝까지 따라가는 예시

아래 점/voxel 수는 **shape 이해용 가상 예시**다. 실행 측정값이 아니다.

```text
frame t=10, B=1
  current_points                         #[1000,3] 현재 robot m
  previous_prediction                    #[1800,3] 이전 robot m
  transformation                         #[4,4]
       ↓ rigid alignment
  aligned_previous                       #[1800,3] 현재 robot m
       ↓ 각각 crop + voxel centroid
  current occupied voxels                #[700,3] C + #[700,3] offsets
  previous occupied voxels               #[1500,3] C + #[1500,3] offsets
       ↓ batch/time 추가, temporal row concatenate
  packed.C                               #[2200,5] int32
  packed.F                               #[2200,3] float32
       ↓ E0..E4: row 수 가변, 채널 32→64→128→256→256
       ↓ D3..D0: 후보 생성→skip 합집합→conv→occupancy→prune 반복
  D0 logits                              #[5000,1]
  keep mask                              #[5000] bool, true=2400
  pruned feature                         #[2400,32]
       ↓ temporal offset head, 겹친 공간 row 축약
  offsets.C                              #[1900,5], time=0
  offsets.F                              #[1900,3], voxel 분수 위치
  confidence                             #[1900]
       ↓ (C_xyz + F)*0.05 + origin
  result.points()[0]                      #[1900,3] 현재 robot m
       ├→ output 저장 / GT 비교 / viewer
       └→ detach → previous_prediction    #[1900,3]
                              frame t=11에서 다시 정렬하여 사용
```

## 13. 구현 확인용 테스트와 관련 문서

| 확인하려는 계약 | 기존 테스트 |
|---|---|
| 좌표 변환 / crop·centroid | [test_geometry.py](../tests/test_geometry.py), [test_voxel.py](../tests/test_voxel.py) |
| 입력 정책 / sparse skip·모델 | [test_inputs.py](../tests/test_inputs.py), [test_sparse_blocks.py](../tests/test_sparse_blocks.py), [test_model.py](../tests/test_model.py) |
| 정렬 feedback / GT loss | [test_mapping.py](../tests/test_mapping.py), [test_loss_targets.py](../tests/test_loss_targets.py) |
| packed I/O / 학습 경로 | [test_packed.py](../tests/test_packed.py), [test_packed_dataset.py](../tests/test_packed_dataset.py), [test_experiment.py](../tests/test_experiment.py), [test_training_io.py](../tests/test_training_io.py), [test_training_duration.py](../tests/test_training_duration.py) |
| export / viewer / realtime / timing | [test_evaluation.py](../tests/test_evaluation.py), [test_evaluation_visualization.py](../tests/test_evaluation_visualization.py), [test_evaluation_realtime.py](../tests/test_evaluation_realtime.py), [test_inference_timing.py](../tests/test_inference_timing.py) |

- 실행 명령 상세: [추론 가이드](INFERENCE_GUIDE.md), [실시간 가이드](REALTIME_INFERENCE_GUIDE.md).
- 데이터 수집 상세: [v3](DATASET_GUIDE/DATASET_V3_GUIDE.md), [v4](DATASET_GUIDE/DATASET_V4_GUIDE.md), [solids](DATASET_GUIDE/DATASET_SOLIDS_GUIDE.md), [전용 검증](VALIDATION_SOLIDS_GUIDE.md).
- 논문과 구현의 차이: [PAPER_NOTES.md](PAPER_NOTES.md). 이 문서는 논문 설명보다 **현재 코드가 실제로 하는 연산**을 우선한다.
