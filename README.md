# CvxNet: input mesh to 50 convex parts

Standalone inference for our **HY3D-Bench-trained, 50-part CvxNet** checkpoint.
Supply an OFF, OBJ, PLY, STL, GLB or GLTF mesh and receive a colored convex
decomposition GLB in the **input mesh's original coordinate system**.

The official CvxNet network is included. No PowerACD code, other local repository,
training dataset, chord cache, SDF, occupancy samples, or pre-existing private Docker
image is needed. Weights are downloaded separately from our private Hugging Face repo.

## Requirements

- Linux x86-64, Python 3.10+ on the host, Git, and Docker.
- An NVIDIA GPU, a CUDA-12.8-compatible driver (R570+ recommended), and
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  configured for `docker run --gpus all`.
- Permission to access the private Hugging Face weights repository. The GitHub
  code repository is public. HF authentication is only needed to download
  weights, never to run inference.
- Allow roughly 40–60 GB for images/build cache on a fresh machine, plus ~390 MiB
  for weights. The first build downloads frameworks and compiles PyTorch3D; this
  can take several minutes or longer. Subsequent runs reuse the images.

Recommended/test target: RTX 5090. The Dockerfiles use public CUDA 12.8 and
[NVIDIA TensorFlow 25.02](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-02.html)
base images. Other NVIDIA architectures must be included when building PyTorch3D;
they are not all tested here.

## 1. Clone and build

The code repository is public; no GitHub authentication is required:

```bash
git clone https://github.com/mvkski/cvxnet-inference.git
cd cvxnet-inference
bash build.sh
```

`build.sh` detects the first GPU's compute capability. To build for a different
machine, set it explicitly; e.g. RTX 5090 is `12.0`, H100 is `9.0`, A100 is `8.0`:

```bash
CUDA_ARCH_LIST='8.0;9.0;12.0' MAX_JOBS=4 bash build.sh
```

Building several architectures takes longer. Set `MAX_JOBS=2` if the compiler runs
out of system RAM. The resulting images are `cvxnet-inference-render:1` and
`cvxnet-inference-model:1`. The frameworks run in separate containers to avoid
TensorFlow/PyTorch dependency and GPU-memory conflicts.

## 2. Download weights

Install the Hugging Face CLI in a small host-side virtual environment (skip this
installation if you already have `hf`):

```bash
python3 -m venv .venv
.venv/bin/pip install 'huggingface_hub==1.2.3'
source .venv/bin/activate
hf auth login
hf download my-little-perceptron/cvxnet-hy3d-50parts --revision 5c4c5d1f47fc28932febcebb6b47c749e4029329 --local-dir weights
python3 verify_weights.py weights
```

Enter a read token with access to that private HF repo when prompted. Do not put
tokens into source files or commit them. `weights/` is ignored by Git and is never
included in the Docker build context. A 401/403/404 from this private repo usually
means your HF account/token does not have access.

The checkpoint prefix is `weights/model.ckpt-1000000` (**not** the `.meta` file).
The `.index` and `.data-00000-of-00001` files are required at runtime. `config.json`
and all reference checksums are supplied; `.meta` is included in the download but
is not imported by the wrapper.

## 3. Run inference on a mesh

```bash
python3 infer.py /absolute/path/to/input_mesh.obj --output-dir results/input_mesh
```

Or process several meshes in the same invocation, sharing model initialization:

```bash
python3 infer.py /path/to/camel.off /path/to/airplane.ply --output-dir results/examples
```

Use `--weights-dir /path/to/downloaded/weights` to keep weights elsewhere, or
`--gpu 1` to select another GPU. `--device cpu` disables CUDA for both stages
(much slower; still uses these Docker images). The launcher does not require
TensorFlow, PyTorch, or Trimesh in your host Python environment.

Output directories must be new or empty: existing results are never overwritten.
Multiple inputs must have distinct filename stems. Paths with spaces are supported;
quote them. Commas/newlines in Docker mount paths are rejected. For `.gltf`, keep
referenced buffers/textures next to the file; prefer self-contained `.glb` when
moving inputs between machines. Scene-node transforms and repeated instances are
applied when loading GLB/GLTF.

## Outputs

For `input_mesh.obj`:

```text
results/input_mesh/
  input_mesh.glb                 # 50 distinctly colored convex components
  input_mesh.stats.json          # per-shape count, provenance and timing breakdown
  summary.json                  # average part count and per-shape timings
  run.json                      # status and actual command wall time
  manifest.json                 # input paths and model configuration
  depth/input_mesh.npy          # ~2 MB normalized depth record
  depth/input_mesh.preprocess.json
  render.log
  inference.log
```

The GLB contains a combined mesh with disconnected colored convex components.
It can be opened in Blender or any GLB viewer. No input files are modified;
input/weight mounts are read-only, and inference containers have networking disabled.

`runtime_seconds` is the sum of mesh loading/normalization, depth rendering,
record writing/reading, network forward pass, and hull/GLB export. It **excludes**
one-time container/Python startup, checkpoint loading, kernel warmup, hashes and
JSON writes. `run.json:command_wall_seconds` includes the launcher/container startup
overhead; use that value for actual single-command latency. Network-only time is
`model_inference_seconds`. No Chamfer/concavity metrics are computed by this repo.

## Exact inference protocol

- Bbox-center the input and uniformly scale its longest extent to 1.0.
- Render 20 fixed Fibonacci-sphere views: radius 1.6, 60° FoV, 224×224 pixels.
- Store camera-space depth in uint16 millimeters, with zero background; pass
  float32 meters in `[batch, 224, 224, 20]` layout to the encoder.
- Use the original ResNet18 CvxNet encoder/decoder: latent size 256, 50 parts,
  50 halfplanes per part, sharpness 75, depth mode, `training=False`.
- Restore the checkpoint at step 1,000,000 and intersect each part's predicted
  halfspaces, capped by a 1.3-wide box around its learned center.
- Undo normalization and export all 50 colored parts.

This is the **same hard-convex export convention as our benchmark wrapper**, not
the upstream soft-occupancy-union marching cubes exporter. It does not merge parts,
ensure zero overlap, or repair input meshes. Different preprocessing/export rules
can change results even with identical weights. Inputs with empty geometry,
nonfinite coordinates or zero extent are rejected.

The wrapper deliberately checks the 50-part model configuration. Changing
`n_parts` to 16 is not a way to get a 16-part output from these weights: that needs
a differently shaped checkpoint.

## Tests and provenance

```bash
python3 -m unittest tests.test_cli -v
docker run --rm --network none --entrypoint python cvxnet-inference-model:1 -m unittest discover -s tests -v
docker run --rm --gpus all --network none --entrypoint python cvxnet-inference-render:1 -m unittest discover -s tests -v
```

The lightweight GitHub Actions tests need no checkpoint, private credentials or
GPU. See [THIRD_PARTY.md](THIRD_PARTY.md) for exact upstream revisions/licensing
and [VALIDATION.md](VALIDATION.md) for integration-test results.

The upstream network source is pinned to TensorFlow Graphics commit
`378e9b7352933ce0a1af4f3331695e0fc590ff75`; its two required Python files are vendored
unchanged. Rendering and export helpers are self-contained in `cvxnet_inference/`.

## Troubleshooting

- **Docker permission denied:** configure your Docker access; do not solve this by
  running the whole launcher as root (that would create root-owned outputs).
- **No CUDA GPUs / could not select device driver:** check `nvidia-smi`, the NVIDIA
  Container Toolkit, and Docker's GPU configuration.
- **No kernel image available:** rebuild the renderer for the target GPU using
  `CUDA_ARCH_LIST`. An image built only for a 5090 cannot run its CUDA extension on
  an A100.
- **Keras / checkpoint-name errors in a custom environment:** use the provided
  TensorFlow image and legacy Keras, not a generic Keras 3 installation.
- **OOM on a very large input:** the renderer retains capacity for all faces in
  each rasterization bin to avoid silently dropping geometry. Simplifying the
  input can reduce memory, but changes the input and potentially the result.
- **Stage failed:** inspect `render.log` or `inference.log` and `run.json`. Rerun
into a fresh output directory after fixing the problem.
- Framework diagnostics are saved in the log files; add `--verbose` to also show
  them in the terminal. Nonfatal CUDA registration/autotuner warnings in this
  TensorFlow image do not by themselves mean inference failed; check `run.json`.
