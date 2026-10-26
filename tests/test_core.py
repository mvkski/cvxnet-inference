import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from cvxnet_inference.common import check_config, write_json
from cvxnet_inference.geometry import color, convex_parts, export_glb
from cvxnet_inference.mesh import load_normalized_mesh


class GeometryTests(unittest.TestCase):
    def test_colors_distinct(self):
        self.assertEqual(len({tuple(color(i)) for i in range(50)}), 50)

    def test_fifty_hulls_and_inverse_transform(self):
        rng = np.random.default_rng(1)
        params = rng.normal(scale=0.3, size=50 * (4 + 3 * 50)).astype(np.float32)
        center = np.array([5., -3., 7.])
        reference = convex_parts(params, np.zeros(3), 1.0)
        parts = convex_parts(params, center, 0.5)
        self.assertEqual(len(parts), 50)
        for original, transformed in zip(reference, parts):
            self.assertTrue(original.is_watertight)
            self.assertGreater(original.volume, 0)
            np.testing.assert_allclose(transformed.vertices, original.vertices * 2 + center)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parts.glb"
            export_glb(parts, path)
            restored = trimesh.load(path, force="mesh", process=False)
            self.assertTrue(np.isfinite(restored.vertices).all())
            self.assertEqual(len(restored.split(only_watertight=False)), 50)
            self.assertEqual(len(np.unique(restored.visual.face_colors, axis=0)), 50)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            convex_parts(np.zeros(10), np.zeros(3), 1.0)

    def test_normalization_and_file_formats(self):
        mesh = trimesh.creation.box(extents=[2, 3, 4])
        mesh.apply_translation([5, -7, 10])
        with tempfile.TemporaryDirectory() as directory:
            for extension in ("off", "obj", "ply", "stl", "glb"):
                path = Path(directory) / f"box.{extension}"
                mesh.export(path)
                vertices, faces, center, scale = load_normalized_mesh(path)
                np.testing.assert_allclose(center, [5, -7, 10])
                self.assertEqual(scale, 0.25)
                self.assertEqual(np.ptp(vertices, axis=0).max(), 1.0)
                self.assertEqual(faces.shape[1], 3)

    def test_scene_instances_and_transforms(self):
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box(), geom_name="box", node_name="first")
        scene.graph.update(frame_to="second", matrix=trimesh.transformations.translation_matrix([4, 0, 0]), geometry="box")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.glb"
            scene.export(path)
            vertices, faces, center, scale = load_normalized_mesh(path)
            self.assertEqual(len(faces), 24)
            np.testing.assert_allclose(center, [2, 0, 0])
            self.assertAlmostEqual(float(scale), 0.2)
            self.assertEqual(np.ptp(vertices, axis=0).max(), 1.0)

    def test_reject_point_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.ply"
            trimesh.points.PointCloud(np.zeros((5, 3))).export(path)
            with self.assertRaises(ValueError):
                load_normalized_mesh(path)


class CommonTests(unittest.TestCase):
    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_config(self):
        config = {"checkpoint_prefix": "model.ckpt-1000000", "model": dict(
            n_parts=50, n_half_planes=50, dims=3, latent_size=256,
            sharpness=75.0, image_input=False, depth_d=20, depth_h=224, depth_w=224)}
        check_config(config)
        config["model"]["n_parts"] = 16
        with self.assertRaises(ValueError):
            check_config(config)
        config["model"]["n_parts"] = 50
        config["checkpoint_prefix"] = "../model"
        with self.assertRaises(ValueError):
            check_config(config)


if __name__ == "__main__":
    unittest.main()
