"""Wall-clock inference timing, including completion of CUDA work."""

from contextlib import contextmanager
import time


class InferenceTiming:
    def __init__(self, device):
        self.device = str(device)
        self.frames = 0
        self.total_s = 0.
        self.last_s = 0.

    def _synchronize(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self):
        self._synchronize()
        started = time.perf_counter()
        yield
        self._synchronize()
        self.last_s = time.perf_counter() - started
        self.total_s += self.last_s
        self.frames += 1

    def summary(self):
        return dict(frames=self.frames, total_inference_s=self.total_s,
                    mean_inference_ms=self.total_s * 1000 / self.frames if self.frames else None,
                    device=self.device, cuda_synchronized=self.device.startswith('cuda'),
                    warmup_frames_excluded=0)
