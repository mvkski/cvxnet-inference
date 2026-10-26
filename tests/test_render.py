import unittest

import numpy as np

try:
    import torch
    from pytorch3d.structures import Meshes
    from cvxnet_inference.render_depth import fibonacci_views, render_depth
    import trimesh
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False


@unittest.skipUnless(HAS_RENDERER, "PyTorch3D is available in the rendering image")
class RenderTests(unittest.TestCase):
    def test_view_count_and_uniform_elevations(self):
        elevations, azimuths = fibonacci_views(20, torch.device("cpu"))
        self.assertEqual(elevations.shape, (20,))
        self.assertEqual(azimuths.shape, (20,))
        z = torch.sin(torch.deg2rad(elevations)).numpy()
        np.testing.assert_allclose(z, np.linspace(.95, -.95, 20), atol=1e-6)

    def test_box_depth_record(self):
        mesh = trimesh.creation.box()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch = Meshes(verts=[torch.tensor(mesh.vertices, dtype=torch.float32, device=device)],
                       faces=[torch.tensor(mesh.faces, dtype=torch.int64, device=device)])
        depth = render_depth(batch)
        self.assertEqual(depth.shape, (20, 224, 224))
        self.assertEqual(depth.dtype, np.uint16)
        self.assertGreater(np.count_nonzero(depth), 10000)
        self.assertGreater(np.count_nonzero(depth == 0), 10000)
        self.assertLess(depth.max(), 2500)


if __name__ == "__main__":
    unittest.main()
