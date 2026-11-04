"""Train/evaluate on DynaMapping-preprocessed sequences with detached feedback."""

import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .inputs import INPUT_POLICIES, TEMPORAL_POLICY, merged_input, point_tensor
from .loss import match_keys, reconstruction_loss, spatial_keys, target_voxels
from .model import ReconstructionUNet4D, UNetConfig
from .training import adam_with_decay, checkpoint_config, log_scalars, save_checkpoint, tensorboard_writer
from .voxel import VoxelGrid, transform_points, voxelize


class SequenceDataset:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        path = self.directory / "dataset.json"
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get("format") not in ("neural-scene-dynamapping-v1", "neural-scene-packed-v1") or self.manifest.get("input_policy") not in INPUT_POLICIES:
            raise ValueError("Expected DynaMapping-preprocessed frame data with a supported input policy")
        grid = self.manifest["grid"]
        self.grid = VoxelGrid(grid["voxel_size"], tuple(grid["shape"]), tuple(grid["origin"]))
        self.splits = {name: [] for name in ("train", "validation", "test")}
        names = set()
        for sequence in self.manifest["sequences"]:
            name = sequence["scene"]
            if name in names or len(sequence["frames"]) < 12:
                raise ValueError("Scenes must be unique across splits and contain >=12 frames")
            names.add(name)
            self.splits[sequence["split"]].append(sequence)
        if not all(self.splits.values()):
            raise ValueError("Train, validation and test must all contain sequences")
        self.cache = OrderedDict()
        self.cache_limit = 512

    def frame(self, sequence, index):
        key = (sequence["scene"], index)
        if key not in self.cache:
            packed = self.manifest['format'] == 'neural-scene-packed-v1'
            path = (self.directory / sequence['shard'] if packed else
                    self.directory / "sequences" / sequence["scene"] / sequence["frames"][index]["file"])
            path = path.resolve()
            if self.directory not in path.parents:
                raise ValueError("Frame path escapes dataset directory")
            if packed:
                from .packed import read_frame
                self.cache[key] = read_frame(path, index)
            else:
                with np.load(path, allow_pickle=False) as data:
                    self.cache[key] = {name: data[name].copy() for name in ("current_points", "target_points", "robot_pose", "transformation", "timestamp_ns")}
            if len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def observable_starts(self, sequence, rollout):
        """A fresh rollout must start with evidence; later blind frames remain.

        Starting from an empty scan and empty recurrent state cannot create any
        sparse convolution candidates and therefore cannot supervise occupancy.
        """
        starts = []
        for index in range(len(sequence["frames"]) - rollout + 1):
            metadata = sequence["frames"][index]
            count = metadata.get("current_points")
            if count is None:
                count = len(self.frame(sequence, index)["current_points"])
            if count:
                starts.append(index)
        return starts


def predict(model, frames, previous, grid, device):
    current = [point_tensor(frame["current_points"], device) for frame in frames]
    aligned = [points.new_empty((0, 3)) if history is None else transform_points(history, frame["transformation"])
               for points, history, frame in zip(current, previous, frames)]
    return model(merged_input(current, aligned, grid, input_policy=model.config.input_policy))


def augmented_frame(frame, mirror_xy, rng, position_noise):
    """Paper position noise and a trajectory-consistent x/y reflection.

    Conjugating the relative pose preserves alignment of predictions expressed
    in mirrored robot coordinates. Validation/test remain unaugmented.
    """
    mirror = np.diag([mirror_xy[0], mirror_xy[1], 1., 1.])
    result = frame.copy()
    rotation = mirror[:3, :3]
    result["current_points"] = np.asarray(frame["current_points"] @ rotation, dtype=np.float32)
    result["target_points"] = np.asarray(frame["target_points"] @ rotation, dtype=np.float32)
    result["transformation"] = mirror @ frame["transformation"] @ mirror
    result["robot_pose"] = mirror @ frame["robot_pose"] @ mirror
    if position_noise:
        noise = rng.uniform(-position_noise, position_noise, result["current_points"].shape).astype(np.float32)
        result["current_points"] = result["current_points"] + noise
    return result


def counts_and_errors(coordinates, offsets, targets, grid):
    device = offsets.device
    gt_coords, gt_offsets = target_voxels(targets, grid, device)
    matched, rows = match_keys(spatial_keys(coordinates.to(device), grid.shape), spatial_keys(gt_coords, grid.shape))
    tp = int(matched.sum().item())
    errors = (offsets[matched] - gt_offsets[rows[matched]]) * grid.voxel_size
    return {"tp": tp, "fp": len(coordinates) - tp, "fn": len(gt_coords) - tp,
        "matched_distance_sum_m": float(torch.linalg.vector_norm(errors, dim=1).sum()),
        "matched_height_sum_m": float(errors[:, 2].abs().sum()), "frames": len(targets)}


def summarize(totals):
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return dict(totals, precision=precision, recall=recall, f1=2*tp/max(2*tp+fp+fn, 1),
        matched_centroid_distance_m=totals["matched_distance_sum_m"] / tp if tp else None,
        matched_height_mae_m=totals["matched_height_sum_m"] / tp if tp else None)


def evaluate(model, dataset, split, device, limit=None, feedback=True, on_frame=None):
    model.eval()
    totals = {"tp": 0, "fp": 0, "fn": 0, "matched_distance_sum_m": 0., "matched_height_sum_m": 0., "frames": 0}
    observed = totals.copy()
    per_scene = {}
    with torch.no_grad():
        for sequence in dataset.splits[split]:
            previous = [None]
            scene = {key: 0 for key in totals}
            for index in range(min(len(sequence["frames"]), limit or len(sequence["frames"]))):
                frame = dataset.frame(sequence, index)
                result = predict(model, [frame], previous, dataset.grid, device)
                target = [point_tensor(frame["target_points"], device)]
                stats = counts_and_errors(result.offsets.C, result.offsets.F, target, dataset.grid)
                for key, value in stats.items():
                    totals[key] += value
                    scene[key] += value
                measurement = voxelize(point_tensor(frame["current_points"], device), dataset.grid)
                coords = torch.cat((torch.zeros((len(measurement.coordinates), 1), dtype=torch.int32, device=device), measurement.coordinates), 1)
                raw_stats = counts_and_errors(coords, measurement.offsets, target, dataset.grid)
                for key, value in raw_stats.items():
                    observed[key] += value
                if on_frame is not None:
                    on_frame(sequence, index, frame, result, summarize(stats), summarize(raw_stats))
                previous = [points.detach() for points in result.points()] if feedback else [None]
            per_scene[sequence["scene"]] = summarize(scene)
    return {"model": summarize(totals), "measurement_baseline": summarize(observed), "per_scene": per_scene}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def calibrate_batch_norm(model, dataset, device, episodes, batch_size, rollout, seed):
    """Re-estimate BN buffers using clean TRAIN rollouts, without labels/gradients."""
    rng = np.random.default_rng(seed)
    windows = [(s, dataset.observable_starts(s, rollout)) for s in dataset.splits['train']]
    windows = [(s, starts) for s, starts in windows if starts]
    if not windows:
        raise ValueError('No observed training rollouts for BN calibration')
    norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    momenta = [norm.momentum for norm in norms]
    used = {sequence['scene']: 0 for sequence, _ in windows}
    model.train()
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None  # cumulative mean across representative batches
    try:
        with torch.no_grad():
            for episode in range(episodes):
                selected = [windows[(episode * batch_size + j) % len(windows)] for j in range(batch_size)]
                starts = [int(rng.choice(starts)) for _, starts in selected]
                previous = [None] * batch_size
                for sequence, _ in selected:
                    used[sequence['scene']] += 1
                for age in range(rollout):
                    frames = [dataset.frame(sequence, start + age) for (sequence, _), start in zip(selected, starts)]
                    result = predict(model, frames, previous, dataset.grid, device)
                    previous = [points.detach() for points in result.points()]
    finally:
        for norm, momentum in zip(norms, momenta):
            norm.momentum = momentum
        model.eval()
    return {'split': 'train only', 'augmentation': 'none', 'episodes': episodes,
            'rollout': rollout, 'batch_size': batch_size, 'normalization_layers': len(norms),
            'scene_episode_counts': used, 'parameters_updated': False}


def snapshot_source(output):
    """Freeze exact experiment code alongside weights; never replace a snapshot."""
    root = Path(__file__).resolve().parents[1]
    destination = output / 'source'
    if destination.exists():
        return
    hashes = {}
    for folder in ('neural_scene', 'scripts', 'tests'):
        for path in sorted((root / folder).glob('*')):
            if not path.is_file():
                continue
            target = destination / folder / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / 'source_hashes.json', hashes)


def save_latest(path, model, optimizer, scheduler, step, rng, metadata):
    temporary = path.with_suffix(".tmp")
    torch.save({"format": "neural-scene-training-v1", "input_policy": model.config.input_policy,
        "model": model.state_dict(), "config": asdict(model.config),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "step": step, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if str(next(model.parameters()).device).startswith("cuda") else None,
        "metadata": metadata}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-policy", choices=INPUT_POLICIES,
        help="New runs default to aligned_temporal_t01; resume/evaluation inherit the saved policy")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rollout", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--position-noise", type=float, default=0.05, help="Uniform observation noise bound in metres, from the paper")
    parser.add_argument("--initial-lr", type=float, default=0.01)
    parser.add_argument("--final-lr", type=float, default=0.0001)
    parser.add_argument("--skip-test", action="store_true", help="Select candidate settings on validation without opening the test split")
    parser.add_argument("--initial-checkpoint", type=Path, help="Warm start weights; starts a new optimizer/run, not resume")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate a frozen checkpoint without optimizer updates")
    parser.add_argument("--calibrate-bn", type=int, default=0, metavar='EPISODES', help='Re-estimate BN on clean train rollouts; use with --evaluate-only')
    parser.add_argument("--no-mirror", action="store_true", help="Disable paper trajectory-wide x/y reflection")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging (enabled by default in OUTPUT alongside checkpoints)")
    args = parser.parse_args()
    if args.evaluate_only and (args.initial_checkpoint is None or args.resume):
        parser.error('--evaluate-only requires --initial-checkpoint and cannot resume training')
    if args.calibrate_bn < 0 or (args.calibrate_bn and not args.evaluate_only):
        parser.error('--calibrate-bn must be nonnegative and requires --evaluate-only')
    if min(args.updates, args.batch_size, args.rollout, args.eval_every) < 1:
        parser.error("Training counts must be positive")
    if not np.isfinite(args.position_noise) or args.position_noise < 0:
        parser.error("--position-noise must be finite and nonnegative")
    if not all(np.isfinite(x) and x > 0 for x in (args.initial_lr, args.final_lr)):
        parser.error('Learning rates must be positive and finite')
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable here; run this command in the GPU-enabled host terminal")
    import MinkowskiEngine as ME
    if str(args.device).startswith("cuda") and not ME.is_cuda_available():
        parser.error("MinkowskiEngine CPU backend selected; use NEURAL_SCENE_DEPS=.deps-cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = SequenceDataset(args.dataset)
    center_xy = np.asarray(dataset.grid.origin[:2]) + np.asarray(dataset.grid.shape[:2]) * dataset.grid.voxel_size / 2
    if not args.no_mirror and not np.allclose(center_xy, 0):
        parser.error("Trajectory mirroring requires a robot-centered XY grid; use --no-mirror for an asymmetric grid")
    if any(len(sequence["frames"]) < args.rollout for sequence in dataset.splits["train"]):
        parser.error("Every training sequence must contain a full rollout")
    train_windows = [(sequence, dataset.observable_starts(sequence, args.rollout)) for sequence in dataset.splits["train"]]
    train_windows = [(sequence, starts) for sequence, starts in train_windows if starts]
    if not train_windows:
        parser.error("No training rollout has an observed first frame")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        parser.error("This training run is complete; use a new output directory")
    if (output / "latest.pt").exists() and not args.resume:
        parser.error("An existing run requires --resume or a new output directory")
    state = torch.load(output / 'latest.pt', map_location=args.device) if args.resume else None
    initial = torch.load(args.initial_checkpoint, map_location=args.device) if args.initial_checkpoint else None
    saved = state if state is not None else initial
    config = checkpoint_config(saved) if saved is not None else UNetConfig(input_policy=args.input_policy or TEMPORAL_POLICY)
    if args.input_policy is not None and args.input_policy != config.input_policy:
        parser.error('Requested input policy differs from checkpoint; start a new run for a different policy')
    model = ReconstructionUNet4D(config).to(args.device)
    initial_checkpoint_hash = None
    if args.initial_checkpoint is not None:
        initial_checkpoint_hash = hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
        if initial.get('format') != 'neural-scene-unet-v1':
            parser.error('Initial checkpoint format/input policy mismatch')
        if checkpoint_config(initial) != model.config:
            parser.error('Initial checkpoint architecture/pruning configuration differs')
        model.load_state_dict(initial['model'])
    optimizer, scheduler = adam_with_decay(model, args.updates, args.initial_lr, args.final_lr)
    metadata = {"dataset_sha256": dataset.sha256, "updates": args.updates, "batch_size": args.batch_size,
        "rollout": args.rollout, "seed": args.seed, "channels": list(model.config.channels),
        "pruning_threshold": model.config.pruning_threshold, "input_policy": model.config.input_policy, "device": args.device,
        "position_noise_m": args.position_noise, "trajectory_mirror_xy": not args.no_mirror,
        "rollout_start_policy": "nonempty first observation; subsequent blind frames retained",
        "train_windows": {sequence["scene"]: len(starts) for sequence, starts in train_windows}}
    metadata.update(initial_lr=args.initial_lr, final_lr=args.final_lr, test_deferred=args.skip_test,
        initial_checkpoint=str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None,
        initial_checkpoint_sha256=initial_checkpoint_hash)
    if args.evaluate_only:
        metadata.update(updates=0, evaluation_only=True)
        if args.calibrate_bn:
            metadata.update(calibration_episodes=args.calibrate_bn, position_noise_m=0., trajectory_mirror_xy=False)
    if (output / "configuration.json").exists() and json.loads((output / "configuration.json").read_text()) != metadata:
        parser.error("Existing output directory uses a different configuration/dataset")
    start_step = 0
    if args.resume:
        if state["metadata"] != metadata:
            parser.error("Resume configuration/dataset differs from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        start_step = state["step"]
        # Remove updates after the last atomic checkpoint so resumed records
        # describe one committed history instead of duplicated replayed steps.
        history_path = output / "training.jsonl"
        if history_path.exists():
            committed = [line for line in history_path.read_text().splitlines()
                         if int(json.loads(line)["step"]) <= start_step]
            temporary = history_path.with_suffix(".tmp")
            temporary.write_text("\n".join(committed) + ("\n" if committed else ""))
            temporary.replace(history_path)
    write_json(output / "configuration.json", metadata)
    snapshot_source(output)
    with tensorboard_writer(output,
            resume_step=start_step if args.resume else None, enabled=not args.no_tensorboard) as writer:
        if args.evaluate_only:
            calibration = calibrate_batch_norm(model, dataset, args.device, args.calibrate_bn,
                args.batch_size, args.rollout, args.seed) if args.calibrate_bn else None
            validation = evaluate(model, dataset, 'validation', args.device)
            test = None if args.skip_test else evaluate(model, dataset, 'test', args.device)
            report = {'status': 'completed', 'mode': 'evaluation', 'configuration': metadata,
                      'validation': validation, 'held_out_test': test}
            if calibration is not None:
                save_checkpoint(output / 'best.pt', model)
                report.update(calibration=calibration, best_validation={'step': 0, 'metrics': validation})
            log_scalars(writer, 'validation', validation, 0)
            if test is not None:
                log_scalars(writer, 'test', test, 0)
            write_json(output / 'report.json', report)
            print('Frozen checkpoint evaluation completed: {}'.format(output / 'report.json'), flush=True)
            return
        baseline_path = output / "initial_validation.json"
        if not baseline_path.exists():
            initial = evaluate(model, dataset, "validation", args.device)
            write_json(baseline_path, initial)
            print("Initial validation F1: {:.4f}, measurement F1: {:.4f}".format(initial["model"]["f1"], initial["measurement_baseline"]["f1"]), flush=True)
        if not args.resume:
            log_scalars(writer, "validation", json.loads(baseline_path.read_text()), 0)
        best_path = output / "best_validation.json"
        best_f1 = json.loads(best_path.read_text())["metrics"]["model"]["f1"] if best_path.exists() else -1
        begun = time.monotonic()
        with (output / "training.jsonl").open("a") as log:
            for step in range(start_step + 1, args.updates + 1):
                model.train()
                optimizer.zero_grad()
                selected_windows = [train_windows[int(rng.integers(len(train_windows)))] for _ in range(args.batch_size)]
                selections = [item[0] for item in selected_windows]
                starts = [possible[int(rng.integers(len(possible)))] for _, possible in selected_windows]
                mirrors = [[1, 1] if args.no_mirror else [int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))] for _ in selections]
                previous = [None] * args.batch_size
                metrics = {"loss": 0., "occupancy": 0., "offset": 0., "output_voxels": 0}
                for age in range(args.rollout):
                    frames = [augmented_frame(dataset.frame(sequence, start + age), mirror, rng, args.position_noise)
                              for sequence, start, mirror in zip(selections, starts, mirrors)]
                    result = predict(model, frames, previous, dataset.grid, args.device)
                    loss = reconstruction_loss(result, [frame["target_points"] for frame in frames])
                    if not torch.isfinite(loss.total):
                        raise RuntimeError("Nonfinite loss at update {}, timestep {}".format(step, age))
                    # Detached recurrence means per-step backward is equivalent to
                    # summing the rollout loss and avoids retaining all 12 graphs.
                    (loss.total / args.rollout).backward()
                    previous = [points.detach() for points in result.points()]
                    for key in ("loss", "occupancy", "offset"):
                        value = loss.total if key == "loss" else getattr(loss, key)
                        metrics[key] += float(value.detach()) / args.rollout
                    metrics["output_voxels"] += len(result.offsets) / args.rollout
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError("Nonfinite or missing gradients at update {}".format(step))
                optimizer.step()
                scheduler.step()
                record = dict(metrics, step=step, lr=optimizer.param_groups[0]["lr"], elapsed_s=time.monotonic()-begun)
                log.write(json.dumps(record) + "\n")
                log.flush()
                log_scalars(writer, "train", {key: value for key, value in record.items() if key != "step"}, step)
                if step == 1 or step % 10 == 0:
                    print("Update {}/{} loss={:.4f} output={:.0f} lr={:.6f}".format(step, args.updates, metrics["loss"], metrics["output_voxels"], record["lr"]), flush=True)
                if step % args.eval_every == 0 or step == args.updates:
                    validation = evaluate(model, dataset, "validation", args.device)
                    write_json(output / ("validation_{:05d}.json".format(step)), validation)
                    log_scalars(writer, "validation", validation, step)
                    if writer is not None:
                        writer.flush()
                    f1 = validation["model"]["f1"]
                    if f1 > best_f1:
                        best_f1 = f1
                        save_checkpoint(output / "best.pt", model)
                        write_json(best_path, {"step": step, "metrics": validation})
                    save_latest(output / "latest.pt", model, optimizer, scheduler, step, rng, metadata)
                    print("Validation F1={:.4f}, best={:.4f}".format(f1, best_f1), flush=True)
        save_checkpoint(output / "last.pt", model)
        best = torch.load(output / "best.pt", map_location=args.device)
        model.load_state_dict(best["model"])
        test = None if args.skip_test else evaluate(model, dataset, "test", args.device)
        report = {"status": "completed", "configuration": metadata, "best_validation": json.loads(best_path.read_text()),
            "initial_validation": json.loads(baseline_path.read_text()), "held_out_test": test,
            "elapsed_s_this_invocation": time.monotonic()-begun,
            "note": "Bounded baseline experiment, not a reproduction of the paper's 200,000-step dataset"}
        if test is not None:
            log_scalars(writer, "test", test, report["best_validation"]["step"])
        write_json(output / "report.json", report)
        if test is None:
            print('Training completed; test deferred. Best validation F1={:.4f}; report={}'.format(best_f1, output / 'report.json'), flush=True)
        else:
            print("Training completed. Held-out test F1={:.4f}; report={}".format(test["model"]["f1"], output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
