import hashlib
from pathlib import Path
import unittest


class VendorTests(unittest.TestCase):
    def test_original_upstream_sources(self):
        root = Path(__file__).resolve().parents[1] / "vendor/tensorflow_graphics/projects/cvxnet/lib"
        if not root.is_dir():
            self.skipTest("Vendored network lives in the inference image, not the rendering image")
        for name, expected in {
            "models.py": "557f058841df0fb2cf7736584ed04d8191e511ad",
            "resnet.py": "fe70c3bc0dfb681dc1f793020714b40d4fcba193",
        }.items():
            data = (root / name).read_bytes()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            self.assertEqual(actual, expected, f"{name} diverged from official CvxNet")


if __name__ == "__main__":
    unittest.main()
