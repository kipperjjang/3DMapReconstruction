"""Generate fresh DynaRaySim surface-GT archives and preprocess with DynaMapping.

Run with DynaRaySim's Python 3.9 environment. All outputs and Warp cache stay
inside this workspace. Source repositories and their existing archives are read-only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DYNARAYSIM = Path(os.environ.get("DYNARAYSIM_ROOT", ROOT.parent / "DynaRaySim"))
MAPPING = Path(os.environ.get("DYNAMAPPING_ROOT", ROOT.parent / "DynaMapping"))
sys.path.insert(0, str(DYNARAYSIM))
sys.path.insert(0, str(MAPPING))
os.environ.setdefault("DYNARAYSIM_WARP_CACHE", str(ROOT / ".build" / "warp-cache"))

from dynamapping import DynaMapping
from dynamapping.readers import DynaRaySimReader
from dynamapping.geometry import inverse, transform_points
from dynaraysim.main.config import resolve
from dynaraysim.main.runner import run
from dynaraysim.data.dataset import Dataset
from dynaraysim.data.ground_truth import load_config as load_gt_config

SCENES = [
    ("01_gymnasium_pedestrians", "train"),
    ("02_indoor_partition_turns", "train"),
    ("03_trapezoid_loop", "train"),
    ("06_indoor_furnished_office", "train"),
    ("07_indoor_furnished_lounge", "train"),
    ("04_outdoor_rough_ground", "validation"),
    ("05_outdoor_obstacle_passage", "test"),
]


def crop(points, origin, shape, voxel_size):
    points = np.asarray(points, dtype=np.float32)
    normalized = (points - origin) / voxel_size
    return points[np.all((normalized >= 0) & (normalized < shape), axis=1)]


def centroid_cloud(points, origin, voxel_size):
    """Reduce GT to one centroid per occupied cell without changing its target."""
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    cells = np.floor((points - origin) / voxel_size).astype(np.int32)
    unique, inverse_rows, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse_rows, points)
    return (sums / counts[:, None]).astype(np.float32)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_scene(scene, split, args):
    config_path = DYNARAYSIM / "examples" / "scenarios" / (scene + ".yaml")
    gt_path = DYNARAYSIM / "examples" / "gt" / (scene + ".yaml")
    archive = args.output / "archives" / scene
    destination = args.output / "sequences" / scene
    completed = destination / "sequence.json"
    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved.get("preprocessing") is None:
            # Compatibility with the first collection started before metadata
            # was added: verify its stored grid and explicit prefix length.
            first = destination / saved["frames"][0]["file"]
            with np.load(first, allow_pickle=False) as data:
                compatible = np.allclose(data["origin"], args.origin)
            if not compatible or len(saved["frames"]) != args.max_frames:
                raise ValueError("Existing preprocessing has incompatible grid/frame count")
            saved["preprocessing"] = {"max_frames": args.max_frames, "origin": list(args.origin)}
            atomic_json(completed, saved)
        if saved.get("preprocessing") != {"max_frames": args.max_frames, "origin": list(args.origin)}:
            raise ValueError("Existing preprocessing uses a different frame limit/origin: {}".format(completed))
        return saved
    if not (archive / "manifest.json").exists():
        if archive.exists():
            raise RuntimeError("Incomplete archive exists; inspect it before resuming: {}".format(archive))
        print("Generating {} with {} + composed surface GT".format(scene, args.device), flush=True)
        settings = load_gt_config(gt_path, "composed")
        result = run(resolve(config_path, args.device), str(archive), ground_truth=settings)
        print("Collection {}: {}".format(scene, result["termination"]), flush=True)
    dataset = Dataset(archive, verify=True)
    reader = DynaRaySimReader(archive)
    mapper = DynaMapping(reader.frequency_hz, reader=reader, history_frames=1)
    destination.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(args.origin, dtype=np.float32)
    shape = np.asarray([64, 64, 64], dtype=np.int32)
    frames = []
    for index in range(min(len(reader), args.max_frames)):
        mapper.get_point_clouds(index)
        alignment = mapper.update()
        gt = dataset.read_gt_window(0, "lidar", index, 1, mode="composed", frame="world", radius_m=4.0)[0]
        if int(gt["timestamp_ns"]) != alignment.current.timestamp_ns:
            raise ValueError("Observation and GT timestamps differ")
        current = crop(alignment.current.points, origin, shape, 0.05)
        target = crop(transform_points(inverse(alignment.current.robot_pose), gt["points"]), origin, shape, 0.05)
        target = centroid_cloud(target, origin, 0.05)
        # Keep raw current counts: averaging current before merging predictions
        # would change the combined centroid weights used by the chosen model.
        output = destination / ("frame_{:05d}.npz".format(index))
        if output.exists():
            raise RuntimeError("Unfinished preprocessing output exists: {}".format(output))
        with output.open("xb") as stream:
            np.savez_compressed(stream, current_points=current, target_points=target,
                robot_pose=alignment.current.robot_pose, transformation=alignment.transformation,
                timestamp_ns=np.int64(alignment.current.timestamp_ns), frame_id=np.int64(alignment.current.frame_id),
                origin=origin, voxel_size_m=np.float32(0.05), point_frame="current_robot",
                input_policy="aligned_merge_t0")
        frames.append({"file": output.name, "current_points": len(current), "target_voxels": len(target)})
        if index % 30 == 0:
            print("{} frame {}/{}: observations={}, target voxels={}".format(scene, index, len(reader), len(current), len(target)), flush=True)
    if len(frames) < 12 or not any(f["target_voxels"] for f in frames):
        raise RuntimeError("A sequence requires >=12 frames and nonempty surface GT")
    record = {"scene": scene, "split": split, "archive": str(archive),
        "preprocessing": {"max_frames": args.max_frames, "origin": list(args.origin)},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "collection_device": dataset.manifest["resolved_config"]["run"]["device"],
        "frequency_hz": reader.frequency_hz, "frames": frames}
    atomic_json(completed, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "training_dataset_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=90, help="Prefix length per scenario; scenes are split as units")
    parser.add_argument("--origin", type=float, nargs=3, default=(-1.6, -1.6, -1.6))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.max_frames < 12:
        parser.error("--max-frames must be >=12")
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = []
    for scene, split in SCENES:
        sequences.append(prepare_scene(scene, split, args))
        atomic_json(args.output / "preparation_progress.json", {"completed_sequences": sequences})
    manifest = {"format": "neural-scene-dynamapping-v1", "input_policy": "aligned_merge_t0",
        "grid": {"voxel_size": 0.05, "shape": [64, 64, 64], "origin": args.origin},
        "split_policy": "whole scenario, no adjacent-frame leakage", "sequences": sequences}
    atomic_json(args.output / "dataset.json", manifest)
    print("Prepared dataset: {}".format(args.output / "dataset.json"), flush=True)


if __name__ == "__main__":
    main()
