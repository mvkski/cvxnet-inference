# Package validation — 2026-10-26

## Environment and build

- Linux x86-64, NVIDIA RTX 5090.
- Both Dockerfiles built successfully from their public CUDA/NGC bases. No local
  `cmpt764`, `cvxnet-train`, `cvxnet-preprocess`, or PowerACD image is a dependency.
- Renderer: Python 3.10, PyTorch 2.7.1+cu128, PyTorch3D 0.7.9 at the pinned commit.
- Model: Python 3.12, NVIDIA TensorFlow 2.17.0 / legacy Keras, NumPy 1.26.4,
  SciPy 1.12.0, Trimesh 4.12.2.
- Downloaded the pinned Hugging Face release into a fresh `weights/` directory;
  checkpoint index, data, meta, pointer and configuration SHA-256 checks all pass.

## Automated tests

14 tests are discovered. In the model image 12 pass, with the two renderer tests
skipped because PyTorch3D is deliberately isolated in the other image. In the
rendering image 13 pass, with the vendored-TensorFlow source test skipped because
those files are deliberately isolated in the model image. All test cases have
therefore been exercised in their applicable environment.

Coverage includes normalization/inverse coordinates, OFF/OBJ/PLY/STL/GLB loading,
scene transforms and repeated instances, point-cloud rejection, depth rendering,
50 valid colored hulls, original upstream source hashes, config checks, paths with
spaces, duplicate-name rejection, Docker mount construction and overwrite protection.

## End-to-end regression against the existing benchmark wrapper

Ran the standalone CLI on the following inputs with the step-1,000,000 checkpoint:

| Input | Format | Exported parts | Depth images vs benchmark |
| --- | --- | ---: | --- |
| VHACD camel | OFF | 50 | Bit-identical |
| VHACD chair | OFF | 50 | Bit-identical |
| PartObjaverse `00200996b8f34f55a2dd2f44d316d107` | GLB | 50 | Bit-identical |

Normalization centers/scales are also identical for all three. Camel and the GLB
have identical exported vertex sets to the earlier benchmark. The chair has a
maximum bidirectional nearest-vertex difference of about **0.0343% of input maximum
extent** from the earlier run, with the same vertex and part counts. Exact
cross-process GPU numerical reproducibility is not promised.

As an additional controlled check, supplied identical network predictions for
these three shapes to both the original and standalone hull exporters: vertices,
faces, and colors match **exactly** for all 150 parts. Thus the refactored export
math itself did not cause the cross-run difference.

Also completed a full CPU-only invocation on the chair, exporting 50 parts, and
a GPU invocation using the newly downloaded HF weights and the default
`--weights-dir` location. No training data, original local source checkout or
network connection is mounted inside either inference container.

This is a packaging/regression test, not a new reconstruction-quality benchmark
or a claim of support for every GPU, mesh format variant, or pathological mesh.
Benchmark assets and generated meshes are intentionally not committed to GitHub.
