#!/usr/bin/env python3
"""Validate the published checkpoint files without TensorFlow or GPU dependencies."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_dir", type=Path, nargs="?", default=Path("weights"))
    args = parser.parse_args()
    expected = json.loads((Path(__file__).parent / "checkpoint_checksums.json").read_text())
    for name, metadata in expected.items():
        path = args.weights_dir / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != metadata["size_bytes"] or digest.hexdigest() != metadata["sha256"]:
            raise SystemExit(f"FAILED: {name}")
        print(f"OK: {name}")


if __name__ == "__main__":
    main()
