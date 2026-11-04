"""Paper optimizer and checkpoint utilities; training duration is explicit."""

from dataclasses import asdict
from contextlib import nullcontext
import math
from pathlib import Path

import torch


def tensorboard_writer(directory, resume_step=None, enabled=True):
    """Flush/close on exit and hide uncommitted events when resuming."""
    if not enabled:
        return nullcontext(None)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requires tensorboard. Install requirements-tensorboard.txt "
            "in the training environment, or pass --no-tensorboard."
        ) from error
    return SummaryWriter(str(directory), flush_secs=10,
        purge_step=resume_step + 1 if resume_step is not None else None)


def log_scalars(writer, prefix, metrics, step):
    """Write finite numeric metrics, including nested evaluation summaries."""
    if writer is None:
        return
    for name, value in metrics.items():
        tag = prefix + "/" + name
        if isinstance(value, dict):
            log_scalars(writer, tag, value, step)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            writer.add_scalar(tag, value, step)


def adam_with_decay(model, total_updates, initial_lr=0.01, final_lr=0.0001):
    """Exponential decay between the paper's endpoints over caller-chosen updates.

    Call optimizer.step(), then scheduler.step() exactly once per update.
    The paper does not specify training duration, batch size, or Adam overrides;
    this helper uses PyTorch's default Adam settings.
    """
    if int(total_updates) != total_updates or total_updates <= 0:
        raise ValueError("total_updates must be a positive integer")
    if initial_lr <= 0 or final_lr <= 0:
        raise ValueError("Learning rates must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(final_lr / initial_lr) ** (1 / total_updates))
    return optimizer, scheduler


def save_checkpoint(path, model):
    """Store both weights and the architecture/input policy needed to load them."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "neural-scene-unet-v1",
        "input_policy": model.config.input_policy,
        "config": asdict(model.config),
        "model": model.state_dict(),
    }, temporary)
    temporary.replace(path)


def checkpoint_config(checkpoint):
    """Validate checkpoint policy and restore legacy configs without a policy field."""
    from .inputs import INPUT_POLICIES
    from .model import UNetConfig
    policy = checkpoint.get('input_policy')
    if checkpoint.get('format') not in ('neural-scene-unet-v1', 'neural-scene-training-v1') or policy not in INPUT_POLICIES:
        raise ValueError('Unsupported checkpoint format/input policy')
    config = checkpoint["config"].copy()
    config["channels"] = tuple(config["channels"])
    config.setdefault('input_policy', policy)
    if config['input_policy'] != policy:
        raise ValueError('Checkpoint policy and model configuration disagree')
    return UNetConfig(**config)


def load_checkpoint(path, device="cpu"):
    """Load inference weights using their saved temporal or legacy input policy."""
    from .model import ReconstructionUNet4D
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('format') != 'neural-scene-unet-v1':
        raise ValueError('Inference requires an inference checkpoint, such as best.pt')
    model = ReconstructionUNet4D(checkpoint_config(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model
