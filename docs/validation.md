# Validation guide

Use this checklist before marking a modernization PR ready for review.

## 1. Locked CPU environment

The project uses `pyproject.toml + uv.lock` as the dependency source of truth.
The default lock resolves the CPU PyTorch wheels.

Install the full optional stack:

```bash
uv sync --all-extras
```

Run the ordinary full-stack suite without slow tests:

```bash
uv run python -m pytest -m "not slow" -v
```

Then run the integration smoke tests:

```bash
uv run python -m pytest \
  tests/test_sweeps.py::test_example_experiment \
  tests/test_sweeps.py::test_hydra_sweep \
  tests/test_sweeps.py::test_optuna_sweep \
  -v
```

## 2. GitHub Actions

The Tests workflow covers:

- core tests on Linux Python 3.10, 3.11, and 3.12;
- core tests on Windows and macOS with Python 3.12;
- the optional vision environment;
- ONNX export/runtime parity and static-quantization coverage on Python 3.10 and 3.12;
- the full optional-dependency environment, including torchao-backed QAT smoke coverage;
- selected Hydra/Optuna integration smoke tests;
- coverage collection.

Code Quality PR runs the repository's pre-commit stack, including Ruff lint and
format checks plus the retained documentation, security, YAML/Markdown,
ShellCheck, and spelling checks.

Dependency or Docker changes also run Docker Smoke. That workflow:

1. builds the core image from the official CUDA-enabled PyTorch base;
2. verifies that CUDA-compiled PyTorch 2.14 and torchvision 0.29 were preserved;
3. runs a CPU `fast_dev_run` inside the CUDA image;
4. builds the default full optional image;
5. imports the optional vision, model-zoo, interpretability, sweeps, and torchao QAT stack.

## 3. CUDA runtime smoke

GitHub-hosted Docker smoke validates that the image contains a CUDA-compiled
PyTorch build, but it does not provide a physical NVIDIA GPU. Before a release
that materially changes CUDA behavior, run at least one real GPU smoke test:

```bash
docker build -t lightning-hydra:dev .

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/logs:/workspace/logs" \
  lightning-hydra:dev \
  python src/train.py trainer=gpu +trainer.fast_dev_run=true
```

The integrated PyTorch 2.14 stack was previously validated on a GeForce GTX 1660
with PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, and CUDA runtime 13.0.

When ONNX Runtime GPU-provider behavior changes, also run the provider matrix on a
physical NVIDIA GPU. Verify that CUDA/TensorRT remain active in the created session;
the benchmark utility rejects silent fallback to a lower-priority provider.

## 4. Feature experiments

The shipped image experiments compose in CI without requiring user data.
End-to-end runs require manifests under the configured `data/` paths:

```bash
uv run train-command experiment=cifar10
uv run train-command experiment=image_classification
uv run train-command experiment=image_multitask
uv run train-command experiment=image_reid
uv run train-command experiment=image_vicreg
```

See the corresponding files in `docs/` for manifest layout and task-specific
settings.

## Ready-to-merge criteria

- [ ] locked core and full optional test matrices are green;
- [ ] Code Quality PR is green;
- [ ] Docker Smoke is green when packaging or Docker files changed;
- [ ] PT2/ONNX export, ORT provider, PTQ, and QAT tests are green when deployment code changed;
- [ ] no unresolved license/provenance concern;
- [ ] original ashleve MIT notice is preserved;
- [ ] a real GPU smoke has been run when CUDA behavior materially changed.
