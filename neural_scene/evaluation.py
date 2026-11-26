"""Export checkpoint evaluation, then replay it using the installed DynaViewer.

Model inference and GUI playback can run in different Python environments.
Playback imports no MinkowskiEngine or model code.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


FORMAT = "neural-scene-evaluation-v1"


def export_evaluation(args):
    import torch
    from .experiment import SequenceDataset, evaluate, write_json
    from .training import load_checkpoint
    from .evaluation_mesh import export_source_mesh
    from .inference_timing import InferenceTiming

    inference_timing = InferenceTiming(args.device)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu explicitly if needed")
    dataset = SequenceDataset(args.dataset)
    if args.scene:
        selected = [s for s in dataset.splits[args.split] if s["scene"] == args.scene]
        if not selected:
            raise ValueError("Scene is absent from the requested split: " + args.scene)
        dataset.splits[args.split] = selected
    model = load_checkpoint(args.checkpoint, args.device).eval()
    # Refuse to mix partial/new evaluation outputs with an earlier run.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": FORMAT, "status": "running", "input_policy": model.config.input_policy,
        "point_frame": "current_robot", "units": "metres",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "dataset": str(dataset.directory), "dataset_sha256": dataset.sha256,
        "split": args.split, "device": args.device, "feedback": not args.no_feedback,
        "limit_per_scene": args.limit, "grid": asdict(dataset.grid),
        "model_config": asdict(model.config), "sequences": [],
    }
    for number, sequence in enumerate(dataset.splits[args.split]):
        entry = {"scene": sequence["scene"], "directory": "scene_{:03d}".format(number), "frames": []}
        export_source_mesh(dataset.directory, sequence, args.output, entry)
        manifest["sequences"].append(entry)
    entries = {s["scene"]: s for s in manifest["sequences"]}
    write_json(args.output / "evaluation.json", manifest)

    def save_frame(sequence, index, frame, result, metrics, baseline):
        entry = entries[sequence["scene"]]
        relative = Path(entry["directory"]) / "frame_{:05d}.npz".format(index)
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        points = result.points()[0].detach().cpu().numpy()
        with destination.open("xb") as stream:
            np.savez_compressed(stream, prediction_points=points,
                confidence=result.confidence.detach().cpu().numpy(),
                current_points=frame["current_points"], target_points=frame["target_points"],
                robot_pose=frame["robot_pose"], transformation=frame["transformation"],
                timestamp_ns=frame["timestamp_ns"], frame_id=np.int64(index))
        entry["frames"].append({"file": relative.as_posix(), "frame_id": index,
            "inference_ms": inference_timing.last_s * 1000,
            "timestamp_ns": int(frame["timestamp_ns"]), "model": metrics,
            "measurement_baseline": baseline})
        print("{} frame {}: {} predictions, F1={:.4f}".format(
            sequence["scene"], index, len(points), metrics["f1"]), flush=True)

    started = time.perf_counter()
    metrics = evaluate(model, dataset, args.split, args.device, limit=args.limit,
                       feedback=not args.no_feedback, on_frame=save_frame,
                       inference_timing=inference_timing)
    timing = inference_timing.summary()
    timing.update(evaluation_total_s=time.perf_counter() - started,
                  scope="predict: input transfer/preprocessing and model forward; excludes dataset reads, metrics and export writes")
    write_json(args.output / "timing.json", timing)
    write_json(args.output / "report.json", dict(metrics, timing=timing))
    manifest["timing"] = timing
    manifest["status"] = "completed"
    write_json(args.output / "evaluation.json", manifest)
    print(json.dumps(dict(metrics, timing=timing), indent=2))
    return metrics


def load_export(directory, scene=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "evaluation.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("status") != "completed":
        raise ValueError("Expected a completed Neural Scene evaluation export")
    sequences = [s for s in manifest["sequences"] if scene is None or s["scene"] == scene]
    if not sequences:
        raise ValueError("No matching scene in evaluation export")
    for sequence in sequences:
        for frame in sequence["frames"]:
            path = (directory / frame["file"]).resolve()
            if directory not in path.parents:
                raise ValueError("Frame path escapes evaluation directory")
    return manifest, sequences


def comparison_snapshot(data, foreground, reference, grid, voxels=False, layers=None):
    """Adapt same-time clouds to named layers and the legacy aligned contract.

    The reference occupies the viewer's orange history slot, but is NOT a
    previous frame or recurrent input. Both clouds are already current-robot XYZ.
    """
    keys = {"prediction": "prediction_points", "observation": "current_points", "target": "target_points"}

    def layer(name):
        points = np.asarray(data[keys[name]], dtype=np.float32)
        fields = dict(points=points, sensor_pose=data["robot_pose"],
                      timestamp_ns=int(data["timestamp_ns"]), frame_id=int(data["frame_id"]))
        if voxels:
            origin = np.asarray(grid["origin"])
            cells = np.floor((points - origin) / grid["voxel_size"])
            cells = cells[((cells >= 0) & (cells < np.asarray(grid["shape"]))).all(axis=1)]
            centers = (np.unique(cells, axis=0) + .5) * grid["voxel_size"] + origin
            fields.update(voxel_centers=centers.astype(np.float32),
                          voxel_size_m=grid["voxel_size"], voxel_grid_pose_world=data["robot_pose"])
        return SimpleNamespace(**fields)

    current = layer(foreground)
    history = []
    if reference != "none":
        other = layer(reference)
        history.append(SimpleNamespace(frame=other, aligned_point_cloud=other.points,
                                       aligned_voxels=getattr(other, "voxel_centers", None)))
    names = layers if layers is not None else [foreground] + ([reference] if reference != "none" else [])
    return SimpleNamespace(current=current, history=history, layers={name: layer(name) for name in names})


def playback(args):
    # Normal installed package import: no sibling-checkout path injection.
    from .evaluation_mesh import load_mesh, resolve_mesh
    from .evaluation_viewer import ComparisonViewer

    manifest, sequences = load_export(args.input, args.scene)
    layers = tuple(dict.fromkeys(args.layers or [args.foreground] +
                                ([args.reference] if args.reference != "none" else [])))
    if args.mesh and args.mesh != "auto" and len(sequences) != 1:
        raise ValueError("An explicit --mesh requires a single scene; select --scene NAME")
    # Resolve before opening a window or creating a recording, so missing geometry fails cleanly.
    meshes = {}
    for sequence in sequences:
        path = resolve_mesh(args.mesh, args.input, manifest, sequence, args.dataset)
        meshes[sequence["scene"]] = load_mesh(path) if path is not None else None
    options = dict(title="Neural Scene | " + ", ".join("GT" if n == "target" else n for n in layers),
                   max_points=args.max_points, point_size=args.point_size, distance=args.distance,
                   show=not args.headless)
    if args.recording:
        if args.recording.exists():
            raise FileExistsError(str(args.recording))
        args.recording.parent.mkdir(parents=True, exist_ok=True)
        options["recording_path"] = str(args.recording)
    opacities = dict(prediction=args.prediction_opacity, observation=args.observation_opacity,
                     target=args.gt_opacity, mesh=args.mesh_opacity)
    window = ComparisonViewer(args.viewer, layers, opacities, **options)
    try:
        while True:
            for sequence in sequences:
                window.set_mesh(meshes[sequence["scene"]])
                for metadata in sequence["frames"]:
                    if window.closed:
                        return
                    started = time.perf_counter()
                    with np.load(args.input / metadata["file"], allow_pickle=False) as data:
                        snapshot = comparison_snapshot(data, args.foreground, args.reference,
                                                       manifest["grid"], args.voxels, layers)
                    window.update(snapshot)
                    if hasattr(window, "canvas"):
                        window.canvas.title = "{} | {} frame {} | F1={:.4f}".format(
                            options["title"], sequence["scene"], metadata["frame_id"], metadata["model"]["f1"])
                    if not args.headless and not window.wait_until(started + 1 / args.fps):
                        return
            if not args.loop:
                break
        if not args.no_hold and not args.headless:
            window.hold()
    finally:
        window.close()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def opacity(value):
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be finite and between 0 (hidden) and 1 (opaque)")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("export", help="Evaluate trained weights and save full-resolution clouds/metrics")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, default=Path("outputs/training_dataset_v2"))
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    evaluate.add_argument("--scene")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--limit", type=positive_int, help="First N frames per scene; default: entire sequence")
    evaluate.add_argument("--no-feedback", action="store_true", help="Ablation: disable previous prediction input")
    play = commands.add_parser("play", help="Replay an export with installed DynaViewer; no model dependencies")
    play.add_argument("--input", type=Path, required=True)
    play.add_argument("--scene")
    play.add_argument("--viewer", choices=("vispy", "rerun"), default="vispy")
    play.add_argument("--foreground", choices=("prediction", "observation", "target"), default="prediction")
    play.add_argument("--reference", choices=("target", "observation", "none"), default="target")
    play.add_argument("--layers", nargs="+", choices=("prediction", "observation", "target"),
                      help="Clouds to display together; overrides foreground/reference")
    for name in ("prediction", "observation", "gt"):
        play.add_argument("--" + name + "-opacity", type=opacity, default=1., help="Layer opacity in [0, 1]")
    play.add_argument("--mesh", nargs="?", const="auto", metavar="PATH",
                      help="Overlay world-space mesh in metres; omit PATH to find bundled/source v3 mesh")
    play.add_argument("--mesh-opacity", type=opacity, default=.25)
    play.add_argument("--dataset", type=Path, help="Relocated source v3 dataset for --mesh auto on old exports")
    play.add_argument("--voxels", action="store_true", help="Show occupied cells using the dataset grid")
    play.add_argument("--fps", type=positive_float, default=10.)
    play.add_argument("--max-points", type=positive_int, default=200000)
    play.add_argument("--point-size", type=positive_float, default=3.)
    play.add_argument("--distance", type=positive_float, default=6.)
    play.add_argument("--loop", action="store_true")
    play.add_argument("--no-hold", action="store_true", help="Close after playback instead of holding the last frame")
    play.add_argument("--recording", type=Path, help="Rerun .rrd output")
    play.add_argument("--headless", action="store_true", help="Record Rerun without opening a window")
    args = parser.parse_args()
    if args.command == "play":
        if (args.recording or args.headless) and args.viewer != "rerun":
            parser.error("--recording and --headless require --viewer rerun")
        if args.headless and (not args.recording or args.loop):
            parser.error("--headless requires --recording and cannot use --loop")
    if args.command == "export":
        export_evaluation(args)
    else:
        playback(args)


if __name__ == "__main__":
    main()
