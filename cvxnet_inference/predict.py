"""Restore the official TensorFlow encoder and export convex GLBs."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_graphics.projects.cvxnet.lib import models

from .common import check_config, write_json
from .geometry import convex_parts, export_glb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    config = check_config(manifest["config"])
    params = dict(config["model"], sample_bbx=1024, sample_surf=1024,
                  weight_overlap=0.1, weight_balance=0.01, weight_center=0.001)
    checkpoint = str(args.weights_dir / config["checkpoint_prefix"])
    depth_dir = args.output_dir / "depth"
    tf.disable_eager_execution()
    startup = time.perf_counter()
    placeholder = tf.placeholder(tf.float32, [None, 224, 224, 20])
    model = models.get_model("multiconvex", SimpleNamespace(**params))
    predicted = model.encode(placeholder, training=False)
    saver = tf.train.Saver()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    checkpoint_step = int(tf.train.load_variable(checkpoint, "global_step"))

    def read_depth(name):
        record = np.load(depth_dir / f"{name}.npy", allow_pickle=False)
        depth = np.transpose(record["depth"].astype(np.float32) / 1000.0, [1, 2, 0])[None]
        return record, depth

    rows = []
    with tf.Session(config=session_config) as session:
        saver.restore(session, checkpoint)
        model_load_seconds = time.perf_counter() - startup
        _, first_depth = read_depth(manifest["inputs"][0]["name"])
        warmup_start = time.perf_counter()
        session.run(predicted, {placeholder: first_depth})
        warmup_seconds = time.perf_counter() - warmup_start
        for index, item in enumerate(manifest["inputs"], 1):
            name = item["name"]
            stats = json.loads((depth_dir / f"{name}.preprocess.json").read_text())
            stats.update(status="failed", checkpoint_step=checkpoint_step,
                         checkpoint_prefix=config["checkpoint_prefix"],
                         output_file=str(Path(manifest["output_dir"]) / f"{name}.glb"),
                         configuration=config)
            try:
                start = time.perf_counter()
                record, depth = read_depth(name)
                stats["depth_record_read_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                vector = session.run(predicted, {placeholder: depth})[0]
                stats["model_inference_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                parts = convex_parts(vector, record["source_center"].astype(np.float64),
                                     float(record["source_scale"]), params["n_parts"], params["n_half_planes"])
                export_glb(parts, args.output_dir / f"{name}.glb")
                stats["convex_export_seconds"] = time.perf_counter() - start
                timing_keys = ["preprocessing_runtime_seconds", "depth_record_write_seconds",
                               "depth_record_read_seconds", "model_inference_seconds", "convex_export_seconds"]
                stats.update(status="success", piece_count=len(parts),
                             runtime_seconds=sum(stats[key] for key in timing_keys))
            except Exception as error:
                stats["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                write_json(args.output_dir / f"{name}.stats.json", stats)
            rows.append(stats)
            print(f'[{index}/{len(manifest["inputs"])}] {name}: {len(parts)} parts; '
                  f'{stats["runtime_seconds"]:.3f}s pipeline; '
                  f'{stats["model_inference_seconds"]:.4f}s network', flush=True)
    summary = dict(status="success", shapes=len(rows), checkpoint_step=checkpoint_step,
                   mean_piece_count=float(np.mean([row["piece_count"] for row in rows])),
                   mean_runtime_seconds=float(np.mean([row["runtime_seconds"] for row in rows])),
                   model_load_seconds=model_load_seconds, model_warmup_seconds=warmup_seconds,
                   timing_policy="Per-shape staged pipeline; excludes container/Python startup, model load, warmup, hashing and JSON writes.")
    for key in ("preprocessing_runtime_seconds", "model_inference_seconds", "convex_export_seconds"):
        summary["mean_" + key] = float(np.mean([row[key] for row in rows]))
    write_json(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
