"""Smoke-test the training model."""

import argparse
import json
import torch

from .inputs import merged_input
from .loss import reconstruction_loss
from .model import ReconstructionUNet4D


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    test = commands.add_parser("smoke", help="full-channel forward, loss and backward on synthetic stairs")
    test.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the current environment")
    print(json.dumps(smoke(args.device), indent=2))


if __name__ == "__main__":
    main()
