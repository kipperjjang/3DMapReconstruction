"""Smoke-test the full architecture or infer from a DynaMapping NPZ frame."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .inputs import merged_input, point_tensor
from .loss import reconstruction_loss
from .model import ReconstructionUNet4D, UNetConfig
from .training import load_checkpoint
from .voxel import VoxelGrid, transform_points


def smoke(device):
    torch.manual_seed(7)
    model = ReconstructionUNet4D().to(device).train()
    x, y = torch.meshgrid(torch.linspace(-1.4, 1.4, 20, device=device), torch.linspace(-1.4, 1.4, 20, device=device), indexing="ij")
    z = (torch.floor((x + 1.4) / 0.4) * 0.12 - 0.4)
    target = torch.stack((x.flatten(), y.flatten(), z.flatten()), 1)
    observed = target[::3] + torch.randn_like(target[::3]) * 0.01
    packed = merged_input([observed], [target[1::3]], input_policy=model.config.input_policy)
    result = model(packed)
    loss = reconstruction_loss(result, [target])
    loss.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not torch.isfinite(loss.total) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Smoke test produced a nonfinite loss/gradient or no gradients")
    return {
        "channels": list(model.config.channels),
        "ordinary_convolutions_per_block": 2,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "input_voxels": len(packed.coordinates),
        "time_coordinates": sorted(set(packed.coordinates[:, -1].tolist())),
        "decoder": [{"stride": s.logits.tensor_stride, "candidates": len(s.logits), "kept": int(s.keep.sum())} for s in result.stages],
        "output_points": len(result.points()[0]),
        "loss": float(loss.total.detach()),
        "backward": "passed",
        "weights": "random initialization; this checks execution, not reconstruction quality",
    }


def infer(args):
    model = load_checkpoint(args.checkpoint, args.device).eval()
    if args.alpha is not None:
        model.config = UNetConfig(model.config.channels, args.alpha, model.config.input_policy)
    with np.load(args.input, allow_pickle=False) as data:
        if "current_points" not in data:
            raise ValueError("Input NPZ must contain DynaMapping's current_points in the current robot frame")
        current = point_tensor(data["current_points"], args.device)
        aligned_previous = current.new_empty((0, 3))
        if args.previous_prediction is not None:
            if "transformation" not in data:
                raise ValueError("DynaMapping transformation is required to align the previous prediction")
            with np.load(args.previous_prediction, allow_pickle=False) as previous:
                aligned_previous = transform_points(point_tensor(previous["points"], args.device), data["transformation"])
    grid = VoxelGrid(origin=tuple(args.origin))
    with torch.no_grad():
        result = model(merged_input([current], [aligned_previous], grid, input_policy=model.config.input_policy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidental replacement of another result.
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, points=result.points()[0].cpu().numpy(), confidence=result.confidence.cpu().numpy(), origin=np.array(grid.origin), voxel_size_m=grid.voxel_size, point_frame="current_robot", input_policy=model.config.input_policy)
    return {"output": str(args.output), "points": len(result.confidence)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    test = commands.add_parser("smoke", help="full-channel forward, loss and backward on synthetic stairs")
    test.add_argument("--device", default="cpu")
    predict = commands.add_parser("infer", help="infer with trained weights from a DynaMapping NPZ")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--previous-prediction", type=Path)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--alpha", type=float)
    predict.add_argument("--origin", nargs=3, type=float, default=(-1.6, -1.6, -1.6), metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the current environment")
    print(json.dumps(smoke(args.device) if args.command == "smoke" else infer(args), indent=2))


if __name__ == "__main__":
    main()
