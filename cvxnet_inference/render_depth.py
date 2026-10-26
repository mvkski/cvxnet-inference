"""Render exactly the 20 depth views used by the trained model."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer,
                               RasterizationSettings, look_at_view_transform)
from pytorch3d.structures import Meshes

from .common import write_json
from .mesh import load_normalized_mesh


def fibonacci_views(count, device):
    index = torch.arange(count, device=device, dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * index / float(count)
    elev = torch.rad2deg(torch.asin(z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    azim = torch.rad2deg(torch.remainder(index * golden_angle, 2.0 * np.pi))
    return elev, azim


def render_depth(mesh):
    elev, azim = fibonacci_views(20, mesh.device)
    settings = RasterizationSettings(
        image_size=(224, 224), blur_radius=0.0, faces_per_pixel=1,
        bin_size=32 if mesh.device.type == "cuda" else 0,
        max_faces_per_bin=int(mesh.num_faces_per_mesh().max().item()),
    )
    rendered = []
    # One view at a time, with enough coarse-bin capacity for the entire mesh.
    for index in range(20):
        rotation, translation = look_at_view_transform(
            dist=1.6, elev=elev[index:index + 1], azim=azim[index:index + 1],
            device=mesh.device,
        )
        cameras = FoVPerspectiveCameras(
            device=mesh.device, R=rotation, T=translation, fov=60.0)
        zbuf = MeshRasterizer(cameras=cameras, raster_settings=settings)(mesh).zbuf[..., 0]
        rendered.append(torch.where(zbuf >= 0.0, zbuf, torch.zeros_like(zbuf)))
    depth = torch.cat(rendered, dim=0)
    return torch.clamp(torch.round(depth * 1000.0), 0, 65535).to(torch.uint16).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check NVIDIA Container Toolkit, or use --device cpu")
    for index, row in enumerate(manifest["inputs"], 1):
        path = Path(row["container_path"])
        stats = {"input_file": row["source_path"], "status": "failed"}
        try:
            # Hashing is excluded from per-shape runtime. A GLTF hash covers the
            # main file only, not externally referenced buffers/textures.
            stats["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            start = time.perf_counter()
            vertices, faces, center, scale = load_normalized_mesh(path)
            stats["mesh_load_normalize_seconds"] = time.perf_counter() - start
            render_start = time.perf_counter()
            mesh = Meshes(verts=[torch.from_numpy(vertices).to(device)],
                          faces=[torch.from_numpy(faces).to(device)])
            depth = render_depth(mesh)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stats["depth_render_seconds"] = time.perf_counter() - render_start
            stats["preprocessing_runtime_seconds"] = time.perf_counter() - start
            record = np.empty((), dtype=np.dtype([
                ("depth", np.uint16, (20, 224, 224)),
                ("source_center", np.float32, (3,)), ("source_scale", np.float32)]))
            record["depth"], record["source_center"], record["source_scale"] = depth, center, scale
            write_start = time.perf_counter()
            destination = args.output_dir / f'{row["name"]}.npy'
            temporary = destination.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, record, allow_pickle=False)
            os.replace(temporary, destination)
            stats["depth_record_write_seconds"] = time.perf_counter() - write_start
            stats.update(status="success", input_vertices=len(vertices), input_faces=len(faces))
        except Exception as error:
            stats["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(args.output_dir / f'{row["name"]}.preprocess.json', stats)
        print(f'[{index}/{len(manifest["inputs"])}] rendered {row["name"]}', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
