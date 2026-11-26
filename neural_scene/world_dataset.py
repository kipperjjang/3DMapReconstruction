"""Index recorded NPZ sequences with world-frame poses for training."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .geometry import pose_matrix, previous_to_current
from .inputs import TEMPORAL_POLICY
from .voxel import VoxelGrid

SENSOR_FORMAT = "neural-scene-sensor-npz-v1"
SPLITS = ("train", "validation", "test")


def read_points(data, name):
    points = data[name]
    if (points.ndim != 2 or points.shape[1] != 3
            or points.dtype.kind != "f" or not np.isfinite(points).all()):
        raise ValueError("{} must contain finite floating-point [N, 3] coordinates".format(name))
    return points


def world_pose(data, name):
    pose = data[name]
    if pose.shape != (7,):
        raise ValueError("{} must have shape [7] (XYZ + XYZW)".format(name))
    return pose_matrix(pose)


def local_cloud(points, robot, grid, sensor=None):
    if sensor is None:
        points = (points - robot[:3, 3]) @ robot[:3, :3]
    else:
        robot_from_sensor = previous_to_current(robot, sensor)
        points = points @ robot_from_sensor[:3, :3].T + robot_from_sensor[:3, 3]
    points = points.astype(np.float32)
    normalized = (points - np.asarray(grid.origin, dtype=np.float32)) / np.float32(grid.voxel_size)
    inside = ((normalized >= 0) & (normalized < np.asarray(grid.shape))).all(axis=1)
    return points[inside]


def read_recorded_frame(path, index, ground_truth):
    try:
        with np.load(path, allow_pickle=False) as data:
            points = read_points(data, "points_sensor")
            robot = world_pose(data, "robot_pose")
            sensor = world_pose(data, "sensor_pose")
            target = read_points(data, "target_points_world") if ground_truth else None
            stamp = data.get("timestamp_ns", np.int64(index))
            if stamp.shape != () or stamp.dtype.kind not in "iu" or not 0 <= int(stamp) <= np.iinfo(np.int64).max:
                raise ValueError("timestamp_ns must be a nonnegative scalar int64-compatible value")
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("Invalid recorded frame {}: {}".format(path, error)) from error
    return points, robot, sensor, target, np.int64(stamp)


class RecordedFrames:
    def __init__(self, supervision, grid):
        source = supervision.get("source")
        self.radius = supervision.get("radius", 0)
        if source not in ("ground_truth", "static_fusion"):
            raise ValueError("Choose recorded dataset supervision: ground_truth or static_fusion")
        if source == "static_fusion" and (type(self.radius) is not int or self.radius < 1):
            raise ValueError("static_fusion requires a positive integer radius")
        self.ground_truth = source == "ground_truth"
        self.grid = grid
        self.read = lru_cache(maxsize=8)(read_recorded_frame)

    def observation(self, points, robot, sensor):
        return local_cloud(points, robot, self.grid, sensor)

    def frame(self, paths, index):
        points, robot, sensor, target, stamp = self.read(paths[index], index, self.ground_truth)
        current = self.observation(points, robot, sensor)
        transform = np.eye(4)
        if index:
            previous = self.read(paths[index - 1], index - 1, self.ground_truth)[1]
            transform = previous_to_current(robot, previous)
        if self.ground_truth:
            target = local_cloud(target, robot, self.grid)
        else:
            start = max(0, index - self.radius)
            stop = min(len(paths), index + self.radius + 1)
            clouds = [current]
            for neighbor in range(start, stop):
                if neighbor != index:
                    observed, _, neighbor_sensor, _, _ = self.read(paths[neighbor], neighbor, False)
                    clouds.append(self.observation(observed, robot, neighbor_sensor))
            target = np.concatenate(clouds)
        return dict(current_points=current, target_points=target, robot_pose=robot,
                    sensor_pose=sensor, transformation=transform, timestamp_ns=stamp)


def index_dataset(directory, fusion_radius):
    grid = VoxelGrid()
    supervised = fusion_radius is None
    supervision = {"source": "ground_truth"} if supervised else {
        "source": "static_fusion", "radius": fusion_radius}
    reader = RecordedFrames(supervision, grid)
    sequences = []
    names = set()
    groups = {split: sorted((directory / split).glob("*/")) for split in SPLITS}
    for split, scenes in groups.items():
        if not scenes:
            raise ValueError("Expected at least one sequence directory under {}".format(directory / split))
        for scene in scenes:
            if scene.name in names:
                raise ValueError("Scene names must be unique across splits: {}".format(scene.name))
            names.add(scene.name)
            files = list(scene.glob("*.npz"))
            if not files or any(not path.stem.isascii() or not path.stem.isdecimal() for path in files):
                raise ValueError("Expected NPZ files with integer timestep names in {}".format(scene))
            files.sort(key=lambda path: int(path.stem))
            if len({int(path.stem) for path in files}) != len(files):
                raise ValueError("Duplicate timestep in {}".format(scene))
            frames = []
            previous_stamp = -1
            for index, path in enumerate(files):
                if directory not in path.resolve().parents:
                    raise ValueError("Frame path escapes dataset directory: {}".format(path))
                points, robot, sensor, _, stamp = read_recorded_frame(path, index, supervised)
                if stamp <= previous_stamp:
                    raise ValueError("Timestamps must increase in timestep order: {}".format(path))
                previous_stamp = stamp
                frames.append({"file": str(path.relative_to(directory)),
                               "current_points": len(reader.observation(points, robot, sensor))})
            sequences.append({"scene": scene.name, "split": split, "frames": frames})
    return {"format": SENSOR_FORMAT, "input_policy": TEMPORAL_POLICY,
            "grid": {"voxel_size": grid.voxel_size, "shape": grid.shape, "origin": grid.origin},
            "supervision": supervision, "sequences": sequences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Dataset containing train, validation and test scene directories")
    parser.add_argument("--static-fusion-radius", type=int,
                        help="Use the current scan and up to this many scans on each side as the target")
    args = parser.parse_args()
    if args.static_fusion_radius is not None and args.static_fusion_radius < 1:
        parser.error("--static-fusion-radius must be positive")
    directory = args.directory.resolve()
    path = directory / "dataset.json"
    if path.exists():
        parser.error("{} already exists".format(path))
    try:
        manifest = index_dataset(directory, args.static_fusion_radius)
        with path.open("x") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(path)


if __name__ == "__main__":
    main()
