"""Hard halfspace convex export, matching the benchmark wrapper."""

import colorsys
import os
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import expit
import trimesh


def color(index):
    rgb = colorsys.hsv_to_rgb((index * 0.618033988749895) % 1.0,
                             0.72 + 0.16 * (index % 2),
                             0.86 + 0.13 * ((index // 2) % 2))
    return np.asarray([*(round(value * 255) for value in rgb), 255], dtype=np.uint8)


def convex_parts(params, center, scale, n_parts=50, n_half_planes=50):
    params = np.asarray(params)
    if params.shape != (n_parts * (4 + 3 * n_half_planes),):
        raise ValueError("Unexpected CvxNet parameter vector size")
    if not np.isfinite(params).all() or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid parameters or normalization")
    translations = params[n_parts:4 * n_parts].reshape(n_parts, 3)
    theta = params[4 * n_parts:].reshape(n_parts, n_half_planes, 3)
    angles = np.tanh(theta[..., :2]) * np.pi
    normals = np.stack([
        np.sin(angles[..., 1]) * np.cos(angles[..., 0]),
        np.sin(angles[..., 1]) * np.sin(angles[..., 0]),
        np.cos(angles[..., 1]),
    ], axis=-1)
    offsets = -(expit(theta[..., 2]) * 0.5)
    parts = []
    for index in range(n_parts):
        translation = translations[index].astype(np.float64)
        part_normals = normals[index].astype(np.float64)
        constants = offsets[index] - part_normals.dot(translation)
        halfspaces = np.column_stack([part_normals, constants])
        # Match our benchmark exporter: cap each part to a 1.3-wide box around
        # its learned center. This is NOT soft-occupancy marching cubes.
        box = []
        for axis in range(3):
            positive = np.zeros(4, dtype=np.float64)
            positive[axis] = 1.0
            positive[3] = -(translation[axis] + 0.65)
            negative = np.zeros(4, dtype=np.float64)
            negative[axis] = -1.0
            negative[3] = translation[axis] - 0.65
            box.extend([positive, negative])
        intersections = HalfspaceIntersection(
            np.vstack([halfspaces, box]), translation, qhull_options="QJ").intersections
        if not np.isfinite(intersections).all():
            raise ValueError(f"Nonfinite convex hull for part {index}")
        hull = ConvexHull(intersections, qhull_options="QJ")
        mesh = trimesh.Trimesh(
            vertices=intersections.astype(np.float64) / scale + np.asarray(center)[None, :],
            faces=np.asarray(hull.simplices, dtype=np.int64), process=False, validate=False)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        mesh.visual.face_colors = np.tile(color(index), (len(mesh.faces), 1))
        parts.append(mesh)
    return parts


def export_glb(parts, path):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.glb")
    # One mesh with disconnected, distinctly colored convex components,
    # exactly as in the benchmark outputs.
    trimesh.util.concatenate(parts).export(temporary, file_type="glb")
    os.replace(temporary, path)
