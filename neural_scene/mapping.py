"""Stateful autoregressive reconstruction using DynaMapping pose alignment."""

from contextlib import nullcontext
from dataclasses import dataclass
import operator

import numpy as np
import torch

from .geometry import pose_matrix, previous_to_current
from .inputs import point_tensor, merged_input
from .voxel import VoxelGrid, transform_points


@dataclass
class PointCloudFrame:
    """Python 3.7-compatible equivalent of DynaMapping's input frame.

    points: [N,3] in robot coordinates, metres.
    robot_pose: T_world_robot, 4x4 or XYZ + XYZW quaternion.
    timestamp_ns: strictly increasing integer nanoseconds within an episode.
    """

    points: object
    robot_pose: object
    timestamp_ns: int
    frame_id: int = 0

    def __post_init__(self):
        self.robot_pose = pose_matrix(self.robot_pose)
        self.timestamp_ns = operator.index(self.timestamp_ns)
        self.frame_id = operator.index(self.frame_id)


@dataclass
class MappingStep:
    reconstruction: object
    points: torch.Tensor
    aligned_previous: torch.Tensor
    transformation: np.ndarray
    dt_s: object
    frame_id: int


class NeuralSceneMapper:
    """Feed the last predicted point cloud back into a single-stream network.

    Accepts PointCloudFrame or an existing DynaMapping frame by duck typing.
    Caller controls model.train()/eval(). reset() starts an independent episode.
    State keeps only one detached prediction; raw scan history is not accumulated.
    """

    def __init__(self, model, frequency_hz, grid=VoxelGrid(), initial_pose=None):
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        self.model = model
        self.frequency_hz = float(frequency_hz)
        self.grid = grid
        self.reset(initial_pose=initial_pose)

    def reset(self, initial_pose=None):
        self.initial_pose = None if initial_pose is None else pose_matrix(initial_pose)
        self.previous_pose = None
        self.previous_timestamp_ns = None
        self.previous_prediction = None
        self.last_step = None

    def update(self, frame, track_grad=False):
        """Align previous prediction, voxelize both times, and reconstruct.

        With track_grad=True, the current result retains gradients for a loss.
        The recurrent state is always detached, as in the paper's training setup.
        Frame order errors are raised before changing the sequence state.
        """
        current_pose = pose_matrix(frame.robot_pose)
        timestamp = operator.index(frame.timestamp_ns)
        frame_id = operator.index(getattr(frame, "frame_id", 0))
        if self.previous_timestamp_ns is not None and timestamp <= self.previous_timestamp_ns:
            raise ValueError("Frame timestamps must increase strictly; reset between episodes")
        device = next(self.model.parameters()).device
        points = point_tensor(frame.points, device)
        initial_pose = current_pose if self.initial_pose is None else self.initial_pose
        transformation = np.eye(4)
        aligned = points.new_empty((0, 3))
        dt_s = None
        if self.previous_prediction is not None:
            transformation = previous_to_current(current_pose, self.previous_pose, initial_pose)
            aligned = transform_points(self.previous_prediction, transformation)
            dt_s = (timestamp - self.previous_timestamp_ns) * 1e-9
        context = nullcontext() if track_grad else torch.no_grad()
        with context:
            packed = merged_input([points], [aligned], self.grid)
            reconstruction = self.model(packed)
            predicted = reconstruction.points()[0]
        step = MappingStep(reconstruction, predicted, aligned, transformation, dt_s, frame_id)
        self.initial_pose = initial_pose
        self.previous_pose = current_pose
        self.previous_timestamp_ns = timestamp
        self.previous_prediction = predicted.detach()
        self.last_step = step
        return step


def rollout_loss(mapper, frames, targets, expected_steps=12, occupancy_weight=1.0, offset_weight=1.0):
    """Compute a detached-feedback sequence loss without stepping an optimizer.

    Default length is the paper's 12 timesteps. The caller supplies already
    augmented observations and robot-frame ground truth for each time.
    """
    from .loss import reconstruction_loss
    if len(frames) != expected_steps or len(targets) != len(frames):
        raise ValueError("Expected {} frames and equally many targets".format(expected_steps))
    mapper.reset()
    losses = []
    for frame, target in zip(frames, targets):
        step = mapper.update(frame, track_grad=True)
        losses.append(reconstruction_loss(step.reconstruction, [target], occupancy_weight, offset_weight))
    return torch.stack([loss.total for loss in losses]).mean(), losses
