# GPU → DynaRaySim → DynaMapping → training

사용자 요청: 새 DynaRaySim 데이터 생성, DynaMapping 전처리, 실제 학습 완료.
순서는 **GPU 문제 확인 및 해결이 먼저**다.

1. Host GPU, driver, device nodes, sp_conv PyTorch의 CUDA 상태를 구분한다.
2. 현재 실행 공간에 GPU가 노출되지 않은 경우 사용자에게 host 진단 결과를 요청한다.
   장치 접근 제한을 우회하지 않는다.
3. CUDA 11.7 + RTX 4090에 맞는 MinkowskiEngine CUDA backend를 별도 workspace
   `.deps-cuda`에 빌드한다. 기존 `.deps` CPU backend와 원본 환경/코드는 보존한다.
4. GPU runtime이 확인된 뒤 DynaRaySim으로 observation + surface GT를 생성한다.
5. 기존 DynaMapping으로 robot-frame observation, pose, frame 간 transform을 전처리한다.
   네트워크 feedback은 직전 prediction이며 raw history를 고정 입력으로 대신하지 않는다.
6. Scene/trajectory 단위 train/validation split으로 학습하고 checkpoint 및 held-out
   reconstruction metrics를 기록한다. 실제 완료와 검증은 결과가 있을 때만 보고한다.

기존 확정 아키텍처: 두 conv/block, 채널 32/64/128/256/256, predictive α=0.5 pruning,
정렬한 직전 reconstruction + 현재 관측을 병합/voxelize하고 모든 입력 t=0.
