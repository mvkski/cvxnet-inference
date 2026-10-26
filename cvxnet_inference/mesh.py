"""Load triangle meshes and retain the inverse normalization transform."""

import numpy as np
import trimesh


def load_normalized_mesh(path):
    loaded = trimesh.load(path, process=False)
    # to_geometry applies scene graph transforms and retains repeated instances.
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input must contain a triangle mesh, not a point cloud or path")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("Empty or nonfinite input mesh")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid face indices")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = ((lower + upper) * 0.5).astype(np.float32)
    extent = float((upper - lower).max())
    if not np.isfinite(extent) or extent <= 1e-8:
        raise ValueError("Degenerate mesh extent")
    scale = np.float32(1.0 / extent)
    return ((vertices - center) * scale).astype(np.float32), faces, center, scale
