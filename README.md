# Lightning-Hydra Template — 2026 fork

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.6-792ee5)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-89b8cd)](https://hydra.cc/)
[![Ruff](https://img.shields.io/badge/lint%20%2B%20format-Ruff-d7ff64)](https://docs.astral.sh/ruff/)
[![Tests](https://github.com/Uwaang/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/Uwaang/lightning-hydra-template/actions/workflows/test.yml)
[![Code quality](https://github.com/Uwaang/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/Uwaang/lightning-hydra-template/actions/workflows/code-quality-main.yaml)

A modernized Lightning + Hydra template for reproducible computer-vision experiments.

This fork keeps the useful experiment orchestration of
[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template),
adds independently reimplemented CV capabilities inspired by
[gorodnitskiy/yet-another-lightning-hydra-template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template),
and updates the development workflow for a 2026 PyTorch stack.

## Design

The repository follows a few deliberate rules:

- keep model, loss, and reusable inference code as plain PyTorch where practical;
- use Lightning primarily as the training orchestration layer;
- keep Hydra composition, but type the stable top-level runtime contract;
- use `pyproject.toml`, `uv.lock`, and Ruff as the default development toolchain;
- keep optional CV dependencies out of the minimal core environment;
- treat export and runtime parity as first-class concerns rather than an afterthought.

The default locked environment uses **CPU PyTorch**. CUDA development currently uses
the Docker runtime so installing the locked CPU wheels cannot accidentally replace a
CUDA-enabled PyTorch installation.

## Quickstart

Install [uv](https://docs.astral.sh/uv/) and synchronize the locked core environment:

```bash
uv sync
```

Run the default MNIST smoke path:

```bash
uv run train-command trainer=cpu +trainer.fast_dev_run=true
```

Run the test suite:

```bash
uv run python -m pytest -m "not slow"
```

Install every optional feature group:

```bash
uv sync --all-extras
```

Or select only what is needed:

```bash
uv sync --extra vision
uv sync --extra model-zoo
uv sync --extra interpretability
uv sync --extra sweeps
uv sync --extra tracking
uv sync --extra onnx
uv sync --extra qat
```

## Training and configuration

Hydra remains the experiment composition layer.

```bash
# default training
uv run train-command

# one GPU inside a CUDA-capable environment
uv run train-command trainer=gpu

# image classification experiment
uv run train-command experiment=image_classification

# override individual values
uv run train-command experiment=image_classification trainer.max_epochs=20
```

The stable top-level runtime options are backed by structured Hydra schemas. Dynamic
component groups such as `model`, `data`, `trainer`, `callbacks`, and `logger`
remain flexible `_target_`-driven configs.

Current experiment examples include:

- generic image classification;
- pretrained image-classification fine-tuning with staged backbone unfreezing and optional EMA;
- CIFAR-10 ResNet-18 validation with bounded qualitative image diagnostics;
- shared-backbone multi-task classification with `CombinedLoader`;
- ReID embeddings with GeM and angular-margin objectives;
- VICReg self-supervised pretraining.

Models that expose the `compile` option compile the underlying plain `nn.Module`
with `torch.compile` rather than compiling the whole Lightning module.

## Data and model stack

The integrated optional CV stack includes:

- manifest-driven labeled and unlabeled image datasets;
- Pillow/OpenCV decoding, torchvision transforms v2, and optional MixUp/CutMix batch augmentation;
- label smoothing plus opt-in EMA and pretrained backbone fine-tuning recipes;
- process-aware HDF5 image storage;
- torchvision and timm classifier/backbone adapters;
- segmentation-models-pytorch integration;
- Focal Loss, ArcFace, SphereFace, CosFace, and VICReg;
- Grad-CAM tooling;
- bounded train/validation/error-gallery image diagnostics;
- JSON/CSV prediction export;
- plain state-dict export and run metadata snapshots;
- dependency-free model complexity reports and Pareto-front analysis.

See the focused documentation under `docs/` for component details, including
[`docs/model-profiling.md`](docs/model-profiling.md) for Params/MAC/FLOP/size profiling and
Pareto analysis, and [`docs/training-recipes.md`](docs/training-recipes.md) for label smoothing,
EMA, pretrained initialization, and staged fine-tuning.

## PyTorch and ONNX export

`src.utils.export_onnx` is the primary portable deployment path. It uses the modern
`torch.export`-based ONNX exporter, serializes the returned `ONNXProgram`, and can
verify numerical parity with ONNX Runtime.

`src.utils.export_pt2` remains available when a PyTorch-native exported graph is useful,
but a `.pt2` artifact is not required for ONNX Runtime deployment.

```bash
uv sync --extra onnx
```

```python
import torch

from src.utils import export_onnx

model = ...  # plain torch.nn.Module
example = torch.randn(1, 3, 224, 224)

onnx_program = export_onnx(model, (example,), "artifacts/model.onnx", verify=True)
```

The ONNX extra is tested on Python 3.10 and 3.12. Static INT8 post-training quantization
and provider-aware runtime benchmarking are also part of the optional ONNX stack. Optional
PT2E QAT uses `torchao==0.18.0` through the separate `qat` extra and can export a converted
QAT graph as QDQ ONNX for ONNX Runtime.

## Runtime benchmarking

Runtime benchmarks are independent from the experiment logger and can compare PyTorch
eager/PT2 paths with ONNX Runtime execution providers. The ORT provider names supported
by the benchmark layer are:

- `cpu` -> `CPUExecutionProvider`;
- `xnnpack` -> `XnnpackExecutionProvider` with CPU fallback;
- `cuda` -> `CUDAExecutionProvider` with CPU fallback;
- `tensorrt` -> `TensorrtExecutionProvider` with CUDA/CPU fallback.

The requested primary provider must remain active after session creation; if ORT drops
that provider during session setup, the benchmark fails instead of being mislabeled.
Normal per-node fallback inside an active provider stack can still occur.

```bash
python scripts/ort_provider_matrix.py artifacts/model.onnx \
  --input-shape 1,3,224,224 \
  --providers cpu xnnpack cuda tensorrt \
  --output artifacts/provider-matrix.json
```

Static INT8 PTQ is available through `src.utils.quantize_onnx_static`; the
`scripts/cifar10_ort_spike.py` helper demonstrates FP32-vs-INT8 accuracy, size, and
CPU-latency comparison.

See [docs/onnx-runtime.md](docs/onnx-runtime.md) for benchmark semantics, environment
requirements, validated examples, and x86/Arm64 deployment recipes.

## Experiment reporting

With a logger configured, training can produce research-oriented reports in addition to
scalar curves: confusion matrices, per-class metrics, sample-level prediction tables,
checkpoints/state dicts, and reproducibility metadata. MLflow runs receive these files as
artifacts. Existing PT2/ONNX/benchmark output directories are publishable without making
deployment export a mandatory part of training.

See [docs/experiment-reporting.md](docs/experiment-reporting.md) for the artifact layout
and reporting controls.

## Docker and CUDA

The CUDA path uses the official PyTorch runtime image and keeps the container process
unprivileged.

```bash
make docker-build
make docker-train
```

See [docs/docker.md](docs/docker.md) for image arguments, bind mounts, and runtime
examples.

Docker dependencies are derived from `uv.lock`. The official CUDA base image owns
`torch` and `torchvision`, while `uv export` installs the rest of the locked graph
without replacing those CUDA-enabled builds. A Docker smoke workflow verifies the image,
the CUDA-compiled PyTorch build, and a CPU fast-dev training pass.

## Development

Common commands are exposed through the Makefile:

```bash
make install
make install-all
make format
make lint
make test
make test-full
make train
```

CI currently covers Python 3.10, 3.11, and 3.12 across Linux, plus Python 3.12 on
Windows and macOS. Separate jobs exercise the optional vision and full-stack
environments, and dependency/Docker changes trigger a CUDA-image smoke build.

## Repository layout

```text
configs/       Hydra config groups and experiment recipes
docs/          focused feature and provenance documentation
scripts/       utility scripts
src/callbacks/ training recipe and fine-tuning callbacks
src/data/      datasets and Lightning data modules
src/models/    plain PyTorch components and Lightning task modules
src/utils/     logging, reproducibility, saving, and export utilities
tests/         unit, integration, and smoke tests
```

## Modernization roadmap

Near-term work:

1. evaluate static type checking after the typed Hydra boundary settles;
2. remove duplicate push/PR CI execution where branch protection still gets equivalent coverage;
3. add optional ExecuTorch recipes only where a validated backend exists.

PT2E QAT is now an optional torchao-backed capability rather than part of the core training
path. ExecuTorch remains a future deployment option only if a validated target needs it.
Fabric, FSDP2/DTensor, and Distributed Checkpoint are not
core template features until there is a real multi-GPU use case and corresponding
runtime validation.

Hydra 1.3 remains pinned while Hydra 1.4 is still a development line. Python/PyTorch
version policy will be revisited after the next stable PyTorch release rather than
tracking prereleases in the core template.

## Provenance

The primary baseline is `ashleve/lightning-hydra-template`. The secondary reference
repository did not declare a software license when the integration review was
performed, so its useful behavior was independently reimplemented instead of vendoring
its source.

See [docs/provenance.md](docs/provenance.md) for the detailed provenance policy and
[docs/feature-parity.md](docs/feature-parity.md) for the integration inventory.

## License

The original Lightning-Hydra-Template notice is preserved:

```text
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
