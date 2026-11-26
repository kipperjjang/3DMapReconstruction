"""Build the existing MinkowskiEngine source into this workspace only.

Run with the target environment's Python. CPU is the default for machines
without an available driver. Pass --cuda to build a GPU-capable backend.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=Path(os.environ.get("MINKOWSKI_SOURCE", root / "submodules" / "MinkowskiEngineCuda13")))
    parser.add_argument("--cuda", action="store_true")
    nvcc = shutil.which("nvcc")
    cuda_home = os.environ.get("NEURAL_SCENE_CUDA_HOME") or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    parser.add_argument("--cuda-home", type=Path, default=Path(cuda_home) if cuda_home else (Path(nvcc).resolve().parent.parent if nvcc else Path("/usr/local/cuda")))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Dependency directory (default: workspace .deps)")
    parser.add_argument("--build-dir", type=Path, help="Isolated workspace build copy")
    parser.add_argument("--cuda-arch", help="TORCH_CUDA_ARCH_LIST, e.g. 8.6+PTX for CUDA 11.7/Ada")
    args = parser.parse_args()
    target = args.build_dir or root / ".build" / ("minkowski-source-cuda" if args.cuda else "minkowski-source")
    if not (args.source / "setup.py").is_file():
        parser.error("--source must be a MinkowskiEngine source checkout")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if target.exists():
        shutil.rmtree(str(target))
    shutil.copytree(str(args.source), str(target), ignore=shutil.ignore_patterns(".git", "build", "*.egg-info", "__pycache__"))
    setup_path = target / "setup.py"
    setup = setup_path.read_text()
    # Upstream setup removes its build directory and uninstalls the existing
    # environment package. Neither operation is appropriate for a local build.
    setup = setup.replace('run_command("rm", "-rf", "build")', "# workspace build starts from a clean copy")
    setup = setup.replace('run_command("pip", "uninstall", "MinkowskiEngine", "-y")', "# preserve environment packages")
    import torch
    if int(torch.__version__.split('.')[0]) >= 2:
        setup = setup.replace('-std=c++14', '-std=c++17')
    setup_path.write_text(setup)
    env = os.environ.copy()
    env["MAX_JOBS"] = str(args.jobs)
    command = [sys.executable, "setup.py", "build", "--blas=openblas"]
    if args.cuda:
        env["CUDA_HOME"] = str(args.cuda_home)
        cccl = args.cuda_home / "include" / "cccl"
        if cccl.is_dir():
            env["CPLUS_INCLUDE_PATH"] = str(cccl) + os.pathsep + env.get("CPLUS_INCLUDE_PATH", "")
            env["CPATH"] = str(cccl) + os.pathsep + env.get("CPATH", "")
        if args.cuda_arch:
            env["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
        command.append("--force_cuda")
    else:
        command.append("--cpu_only")
    subprocess.check_call(command, cwd=str(target), env=env)
    libraries = list((target / "build").glob("lib.*"))
    if len(libraries) != 1:
        raise RuntimeError("Expected one built library directory; got {}".format(libraries))
    deps = (root / ".deps" if args.output is None else args.output).resolve()
    deps.mkdir(parents=True, exist_ok=True)
    for name in ("MinkowskiEngine", "MinkowskiEngineBackend"):
        destination = deps / name
        if destination.exists():
            shutil.rmtree(str(destination))
        shutil.copytree(str(libraries[0] / name), str(destination))
    shutil.copyfile(str(args.source / "LICENSE"), str(deps / "MINKOWSKI_LICENSE"))
    print("Built {} backend at {}".format("CUDA" if args.cuda else "CPU", deps))
    print("Set PYTHONPATH={} when using {}.".format(deps, sys.executable))


if __name__ == "__main__":
    main()
