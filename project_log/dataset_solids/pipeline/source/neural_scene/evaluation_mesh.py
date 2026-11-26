"""World-space source meshes for evaluation playback (no model dependencies)."""

import hashlib
from pathlib import Path

import numpy as np


def load_mesh(path):
    """Read authored OBJ or an exported NPZ; other formats need trimesh."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            vertices, faces = data["vertices"].copy(), data["faces"].copy()
    elif path.suffix.lower() == ".obj":
        vertices, faces = [], []
        with path.open() as stream:
            for line in stream:
                fields = line.partition("#")[0].split()
                if not fields:
                    continue
                if fields[0] == "v":
                    vertices.append([float(v) for v in fields[1:4]])
                elif fields[0] == "f":
                    indices = [int(v.split("/")[0]) for v in fields[1:]]
                    if len(indices) < 3 or 0 in indices:
                        raise ValueError("Invalid OBJ face in {}".format(path))
                    indices = [i - 1 if i > 0 else len(vertices) + i for i in indices]
                    faces.extend((indices[0], indices[i], indices[i + 1])
                                 for i in range(1, len(indices) - 1))
    else:
        try:
            import trimesh
        except ImportError as error:
            raise ImportError("Non-OBJ meshes require trimesh; or supply a world-space OBJ") from error
        mesh = trimesh.load(str(path), force="mesh", process=False)
        vertices, faces = mesh.vertices, mesh.faces
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Expected finite mesh vertices and valid triangle indices: {}".format(path))
    return vertices, faces.astype(np.uint32)


def source_mesh_path(dataset, sequence):
    """Find v3's composed, authored world mesh beside its packed shard."""
    if not sequence.get("shard"):
        return None
    root = Path(dataset).resolve()
    candidate = (root / sequence["shard"]).resolve().parent / "scene.obj"
    if root not in candidate.parents:
        raise ValueError("Mesh path escapes dataset directory")
    if not candidate.is_file():
        return None
    expected = sequence.get("geometry_sha256")
    if expected and hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
        raise ValueError("Source mesh hash differs from collected GT: {}".format(candidate))
    return candidate


def export_source_mesh(dataset, sequence, output, entry):
    """Bundle geometry so playback survives moving or removing the dataset."""
    source = source_mesh_path(dataset, sequence)
    if source is None:
        return
    vertices, faces = load_mesh(source)
    relative = Path(entry["directory"]) / "source_mesh.npz"
    destination = Path(output) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, vertices=vertices, faces=faces)
    entry["mesh"] = {"file": relative.as_posix(), "point_frame": "world", "units": "metres",
                     "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def resolve_mesh(mesh, export, manifest, sequence, dataset=None):
    """Resolve an explicit mesh, bundled geometry, or a legacy v3 dataset."""
    if mesh is None:
        return None
    if mesh != "auto":
        return Path(mesh).expanduser().resolve()
    if sequence.get("mesh"):
        root = Path(export).resolve()
        metadata = sequence["mesh"]
        if metadata.get("point_frame") != "world" or metadata.get("units") != "metres":
            raise ValueError("Expected source mesh in world coordinates, metres")
        path = (root / metadata["file"]).resolve()
        if root not in path.parents:
            raise ValueError("Mesh path escapes evaluation directory")
        return path
    root = Path(dataset or manifest.get("dataset", "")).expanduser().resolve()
    index = root / "dataset.json"
    if index.is_file():
        import json
        data = json.loads(index.read_text())
        for source in data["sequences"]:
            if source["scene"] == sequence["scene"]:
                path = source_mesh_path(root, source)
                if path is not None:
                    return path
    raise FileNotFoundError("No source mesh for {}. For an old export, use --dataset PATH "
                            "with --mesh auto, or --mesh WORLD_MESH.obj --scene NAME".format(sequence["scene"]))


def mesh_in_robot(vertices, robot_pose):
    """T_world_robot maps robot points to world; invert it for the mesh."""
    pose = np.asarray(robot_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4x4 robot pose")
    world_to_robot = np.linalg.inv(pose)
    return (np.asarray(vertices) @ world_to_robot[:3, :3].T
            + world_to_robot[:3, 3]).astype(np.float32)
