# Third-party code and reproducibility

The following files are copied byte-for-byte from the official TensorFlow
Graphics CvxNet implementation at commit
`378e9b7352933ce0a1af4f3331695e0fc590ff75`:

- `vendor/tensorflow_graphics/projects/cvxnet/lib/models.py`
- `vendor/tensorflow_graphics/projects/cvxnet/lib/resnet.py`

Upstream: <https://github.com/tensorflow/graphics/tree/378e9b7352933ce0a1af4f3331695e0fc590ff75/tensorflow_graphics/projects/cvxnet>

These files retain the TensorFlow Authors' copyright and Apache-2.0 notices.
The full Apache-2.0 license is included in `LICENSE`. Empty namespace initializers
are supplied here so only the two required source modules need to be vendored.
The unit tests check their original Git blob hashes.

The depth renderer, hard-halfspace exporter, and launcher are standalone wrappers
adapted from our CvxNet benchmark pipeline. They do not change the neural network.
In particular, the hard-hull export is **not** the upstream soft-union marching
cubes extraction. Our renderer preserves the benchmark's fixed Fibonacci views,
normalization, camera convention and millimeter depth quantization. Trimesh replaces
the training helper's Open3D mesh loader; no training/occupancy dependencies are needed.

Build-time dependencies are fetched from their upstream distributions, not vendored:

- PyTorch 2.7.1, torchvision 0.22.1 (CUDA 12.8 wheels).
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) commit
  `1f7f85c0a3abcce0644f33d32ee911f10a9cac26` (0.7.9), matching our benchmark renderer.
- NVIDIA TensorFlow container `25.02-tf2-py3` (TensorFlow 2.17.0, legacy Keras).
- NumPy, SciPy, Trimesh and dependencies listed in the requirements files.

Dependencies and NVIDIA containers retain their respective licenses/terms.
The weights are a separately trained HY3D-Bench checkpoint, not an official
pretrained CvxNet release. No third-party dataset meshes are distributed here.

Please cite the original method when using CvxNet:

> Boyang Deng, Kyle Genova, Soroosh Yazdani, Sofien Bouaziz, Geoffrey Hinton,
> and Andrea Tagliasacchi. CvxNet: Learnable Convex Decomposition. CVPR 2020.
