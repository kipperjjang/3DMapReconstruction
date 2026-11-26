"""Read-only GPU preflight; distinguishes missing device exposure from CUDA setup."""

import argparse
import glob
import json
import os
from pathlib import Path
import stat
import subprocess

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    try:
        nvidia = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        nvidia_status = {"returncode": nvidia.returncode, "output": (nvidia.stdout + nvidia.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        nvidia_status = {"error": str(error)}
    available = torch.cuda.is_available()
    report = {
        "device_nodes": sorted(path for path in glob.glob("/dev/nvidia*") if stat.S_ISCHR(os.stat(path).st_mode)),
        "torch": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": available,
        "device_count": torch.cuda.device_count(),
        "nvidia_smi": nvidia_status,
    }
    proc_gpus = Path("/proc/driver/nvidia/gpus")
    if proc_gpus.exists():
        report["kernel_gpu_info"] = {p.parent.name: p.read_text() for p in proc_gpus.glob("*/information")}
    if available:
        report["device"] = torch.cuda.get_device_name(0)
        try:
            x = torch.randn((256, 256), device="cuda", requires_grad=True)
            (x @ x.T).square().mean().backward()
            torch.cuda.synchronize()
            report["torch_cuda_forward_backward"] = "passed"
        except RuntimeError as error:
            report["torch_cuda_forward_backward"] = str(error)
            available = False
    print(json.dumps(report, indent=2))
    if args.require_cuda and not available:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
