import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import infer


class CLITests(unittest.TestCase):
    def test_paths_with_spaces_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "some shape.off"
            path.touch()
            items, mounts = infer.prepare_inputs([path])
            self.assertEqual(items[0]["container_path"], "/inputs/0/some shape.off")
            self.assertEqual(mounts, {root: "/inputs/0"})
            with self.assertRaises(ValueError):
                infer.prepare_inputs([path, path])

    def test_comma_mount_rejected(self):
        with self.assertRaises(ValueError):
            infer.checked_mount("/tmp/a,b", "/inputs/0")

    def test_pipeline_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "some shape.off"
            mesh.touch()
            weights = root / "weights"
            weights.mkdir()
            config = json.loads((infer.ROOT / "model_config.json").read_text())
            (weights / "config.json").write_text(json.dumps(config))
            for suffix in (".index", ".data-00000-of-00001"):
                (weights / (config["checkpoint_prefix"] + suffix)).touch()
            output = root / "output with spaces"
            with patch("infer.subprocess.run"), patch("infer.shutil.which", return_value="docker"), patch("infer.execute") as execute:
                infer.main([str(mesh), "--weights-dir", str(weights), "--output-dir", str(output)])
                self.assertEqual(execute.call_count, 2)
                render = execute.call_args_list[0].args[0]
                prediction = execute.call_args_list[1].args[0]
                self.assertIn("device=0", render)
                self.assertIn("none", render)
                self.assertIn(f"type=bind,src={weights},dst=/weights,readonly", prediction)
                self.assertNotIn(str(infer.ROOT), render)
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "success")
            with self.assertRaises(ValueError):
                infer.main([str(mesh), "--weights-dir", str(weights), "--output-dir", str(output)])


if __name__ == "__main__":
    unittest.main()
