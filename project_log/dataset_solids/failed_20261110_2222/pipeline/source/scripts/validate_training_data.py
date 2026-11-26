"""Check every prepared frame before GPU training (NumPy only)."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scene_geometry", ROOT / "neural_scene" / "geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def validate(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / "dataset.json").read_text())
    assert manifest["format"] == "neural-scene-dynamapping-v1"
    assert manifest["input_policy"] == "aligned_merge_t0"
    grid = manifest["grid"]
    origin = np.asarray(grid["origin"], dtype=np.float32)
    shape = np.asarray(grid["shape"])
    voxel_size = np.float32(grid["voxel_size"])
    scenes, splits = set(), {split: 0 for split in ("train", "validation", "test")}
    report = {"status": "validated", "split_frames": splits, "sequences": []}
    if "max_dataset_bytes" in manifest:
        size = sum(max(p.stat().st_size, p.stat().st_blocks * 512)
                   for p in directory.rglob('*') if p.is_file())
        assert size < manifest['max_dataset_bytes'], 'Dataset exceeds byte budget'
        report.update(dataset_bytes=size, max_dataset_bytes=manifest['max_dataset_bytes'])
    for sequence in manifest["sequences"]:
        name = sequence["scene"]
        assert name not in scenes, "Scene appears in multiple splits"
        scenes.add(name)
        assert len(sequence["frames"]) >= 12
        previous_pose = initial_pose = previous_time = None
        counts, targets, max_error = [], [], 0.
        for index, metadata in enumerate(sequence["frames"]):
            path = (directory / "sequences" / name / metadata["file"]).resolve()
            assert directory in path.parents, "Frame escapes dataset directory"
            if 'sha256' in metadata:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256'], (name, index, 'frame hash mismatch')
            with np.load(path, allow_pickle=False) as frame:
                assert str(frame["input_policy"]) == "aligned_merge_t0"
                assert str(frame["point_frame"]) == "current_robot"
                np.testing.assert_allclose(frame["origin"], origin, atol=0, rtol=0)
                np.testing.assert_allclose(frame["voxel_size_m"], voxel_size, atol=0, rtol=0)
                for field, count in (("current_points", "current_points"), ("target_points", "target_voxels")):
                    points = frame[field]
                    assert points.ndim == 2 and points.shape[1] == 3
                    assert np.isfinite(points).all(), (name, index, field)
                    cells = np.floor((points - origin) / voxel_size).astype(np.int32)
                    assert np.all((cells >= 0) & (cells < shape)), (name, index, "outside grid")
                    assert len(points) == metadata[count]
                    if field == "target_points":
                        assert len(np.unique(cells, axis=0)) == len(points), (name, index, "duplicate GT cell")
                counts.append(len(frame["current_points"]))
                targets.append(len(frame["target_points"]))
                timestamp = int(frame["timestamp_ns"])
                assert previous_time is None or timestamp > previous_time
                pose = geometry.pose_matrix(frame["robot_pose"])
                if initial_pose is None:
                    initial_pose = pose
                expected = np.eye(4) if previous_pose is None else geometry.previous_to_current(pose, previous_pose, initial_pose)
                error = float(np.max(np.abs(frame["transformation"] - expected)))
                assert error < 1e-8, (name, index, "feedback transform mismatch", error)
                max_error = max(max_error, error)
                previous_pose, previous_time = pose, timestamp
        splits[sequence["split"]] += len(counts)
        report["sequences"].append({"scene": name, "split": sequence["split"], "frames": len(counts),
            "blind_frames": sum(count == 0 for count in counts),
            "observation_count_min_max": [min(counts), max(counts)],
            "target_voxels_min_max": [min(targets), max(targets)],
            "max_feedback_transform_error": max_error})
    assert all(splits.values()), "Missing dataset split"
    report["total_frames"] = sum(splits.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs" / "training_dataset_v1")
    args = parser.parse_args()
    report = validate(args.dataset)
    path = args.dataset / "validation_report.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
