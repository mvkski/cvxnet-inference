"""Framework-independent file and input helpers."""

import json
import os
import uuid
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def check_config(config):
    """Fail explicitly instead of silently running a mismatched checkpoint."""
    model = config["model"]
    expected = dict(n_parts=50, n_half_planes=50, dims=3, latent_size=256,
                    sharpness=75.0, image_input=False, depth_d=20,
                    depth_h=224, depth_w=224)
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"This release requires model.{key}={value!r}; got {model.get(key)!r}")
    prefix = config["checkpoint_prefix"]
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("checkpoint_prefix must be a filename, without directories")
    return config
