"""Compare alignment to the existing DynaMapping implementation, read-only.

Run using /home/intern/DynaRaySim/.venv/bin/python. No torch is needed here.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scene_geometry", root / "neural_scene" / "geometry.py")
scene_geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene_geometry)
sys.path.insert(0, "/home/intern/DynaRaySim/DynaMapping")
from dynamapping import DynaMapping, PointCloudFrame


def main():
    random = np.random.default_rng(17)
    mapper = DynaMapping(10)
    previous = None
    max_error = 0.0
    for index in range(100):
        pose = np.concatenate((random.normal(size=3), random.normal(size=4)))
        frame = PointCloudFrame(random.normal(size=(10, 3)), pose, index * 100000000, index)
        result = mapper.update(frame)
        if previous is not None:
            ours = scene_geometry.previous_to_current(frame.robot_pose, previous.robot_pose, mapper.initial_pose)
            np.testing.assert_allclose(ours, result.transformation, atol=1e-12)
            max_error = max(max_error, float(np.max(np.abs(ours - result.transformation))))
        previous = frame
    print("99 random pose-pair comparisons to DynaMapping.update() passed; max error:", max_error)


if __name__ == "__main__":
    main()
