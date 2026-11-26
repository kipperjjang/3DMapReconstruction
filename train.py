"""Edit TrainConfig and run this file from the project root."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from neural_scene.experiment import run_experiment


@dataclass
class TrainConfig:
    # Dataset directory must contain dataset.json. Use a new output for each run.
    dataset: Path = Path("logs")
    output: Path = Path("outputs/sensor_training")
    device: str = "cuda:0"
    input_policy: Optional[str] = None

    # Choose one duration. Both None uses the experiment default of 500 updates.
    updates: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: int = 2
    rollout: int = 12
    eval_every: int = 50
    seed: int = 20261011
    position_noise: float = 0.05
    no_mirror: bool = False
    initial_lr: float = 0.01
    final_lr: float = 0.0001

    no_tensorboard: bool = False
    skip_test: bool = False
    resume: bool = False
    initial_checkpoint: Optional[Path] = None
    evaluate_only: bool = False
    calibrate_bn: int = 0


def main(config=None):
    run_experiment(TrainConfig() if config is None else config)


if __name__ == "__main__":
    main()
