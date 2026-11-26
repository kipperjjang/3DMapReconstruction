# Progress

- Added realtime CLI, held-key and SDL joystick commands, sensor-frame to robot-frame
  conversion, scene collision checks, fresh scans and recurrent model adapter.
- Added live mesh/observation/prediction visualization with existing opacity controls.
- Scene switch/reset constructs a fresh simulator and mapper; no archived scans are used.
- CLI help imports successfully via scripts/run_dyna.sh in conda dyna.
- `RUN_SIM_TESTS=1 bash scripts/run_dyna.sh -m pytest tests/test_evaluation_realtime.py tests/test_mapping.py -q`
  passed all 7 tests, including CPU fresh raycast, ego motion, joystick deadzone,
  disconnection and existing mapper feedback checks. `git diff --check` passed.
- dyna currently cannot import MinkowskiEngine through run_dyna.sh. Actual checkpoint
  inference remains unverified until its compiled extension is on PYTHONPATH.
- GUI and hardware validation deferred per user instruction. No GUI opened.
