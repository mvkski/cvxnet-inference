#!/usr/bin/env python3
"""Run mesh -> 20 depth views -> CvxNet -> colored convex GLB, using Docker."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cvxnet_inference.common import check_config, write_json

ROOT = Path(__file__).resolve().parent
SUPPORTED = {".off", ".obj", ".ply", ".stl", ".glb", ".gltf"}


def prepare_inputs(inputs):
    records, mounts, names = [], {}, set()
    for path in inputs:
        path = path.expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Expected a mesh file ({', '.join(sorted(SUPPORTED))}): {path}")
        if path.stem in names:
            raise ValueError(f"Duplicate output name {path.stem!r}; rename one input or run separately")
        names.add(path.stem)
        if path.parent not in mounts:
            mounts[path.parent] = f"/inputs/{len(mounts)}"
        records.append(dict(name=path.stem, source_path=str(path),
                            container_path=str(Path(mounts[path.parent]) / path.name)))
    return records, mounts


def checked_mount(source, destination, readonly=False):
    # Docker --mount uses comma-delimited fields; spaces are safe as separate argv.
    if any(char in str(source) for char in (",", "\n", "\r")):
        raise ValueError(f"Docker mount paths must not contain commas or newlines: {source}")
    return f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")


def execute(command, log_path, verbose=False):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                # Keep verbose framework/compiler diagnostics in the log by
                # default; print our per-shape progress without hundreds of
                # CUDA-autotuner warnings obscuring the output paths.
                if verbose or line.startswith("["):
                    print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        print("".join(log_path.read_text().splitlines(keepends=True)[-30:]), file=sys.stderr)
        raise RuntimeError(f"Stage failed (exit {status}); see {log_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", type=Path, nargs="+", help="OFF, OBJ, PLY, STL, GLB or GLTF")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", default="0", help="Docker GPU index/UUID; ignored with --device cpu")
    parser.add_argument("--render-image", default="cvxnet-inference-render:1")
    parser.add_argument("--model-image", default="cvxnet-inference-model:1")
    parser.add_argument("--verbose", action="store_true", help="Also show framework diagnostics in the terminal")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    records, input_mounts = prepare_inputs(args.meshes)
    weights = args.weights_dir.expanduser().resolve(strict=True)
    config_path = weights / "config.json"
    # Bundled config permits using an existing local checkpoint directory.
    if not config_path.is_file():
        config_path = ROOT / "model_config.json"
    config = check_config(json.loads(config_path.read_text()))
    prefix = weights / config["checkpoint_prefix"]
    for suffix in (".index", ".data-00000-of-00001"):
        if not Path(str(prefix) + suffix).is_file():
            raise ValueError(f"Missing checkpoint file: {prefix}{suffix}; see README download instructions")
    output = args.output_dir.expanduser().resolve()
    if output == weights or output in weights.parents or weights in output.parents:
        raise ValueError("Keep output and weights directories separate")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing to overwrite a nonempty output directory: {output}")
    if shutil.which("docker") is None:
        raise RuntimeError("Docker not found; install Docker and NVIDIA Container Toolkit first")
    for image in (args.render_image, args.model_image):
        subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)

    common = ["docker", "run", "--rm", "--network", "none", "--ipc=host",
              "--user", f"{os.getuid()}:{os.getgid()}",
              "--env", "XDG_CACHE_HOME=/tmp/cache", "--env", "PYTHONUNBUFFERED=1",
              "--mount", checked_mount(output, "/work")]
    if args.device == "cuda":
        common += ["--gpus", f"device={args.gpu}"]
    else:
        common += ["--env", "CUDA_VISIBLE_DEVICES=-1"]
    render = list(common)
    for source, destination in input_mounts.items():
        render += ["--mount", checked_mount(source, destination, readonly=True)]
    render += [args.render_image, "--manifest", "/work/manifest.json",
               "--output-dir", "/work/depth", "--device", "cuda:0" if args.device == "cuda" else "cpu"]
    predict = common + ["--mount", checked_mount(weights, "/weights", readonly=True),
                        args.model_image, "--manifest", "/work/manifest.json",
                        "--weights-dir", "/weights", "--output-dir", "/work"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", dict(inputs=records, output_dir=str(output), config=config))
    state = dict(status="running", render_image=args.render_image, model_image=args.model_image,
                 device=args.device, shapes=len(records))
    try:
        print("Rendering depth images...", flush=True)
        execute(render, output / "render.log", args.verbose)
        print("Running CvxNet and exporting convex hulls...", flush=True)
        execute(predict, output / "inference.log", args.verbose)
        state["status"] = "success"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["command_wall_seconds"] = time.perf_counter() - start
        write_json(output / "run.json", state)
    for item in records:
        print(output / (item["name"] + ".glb"))
    print(f"Stats: {output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
