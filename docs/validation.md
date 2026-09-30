# Validation guide

Use this checklist before marking the full integration PR ready for review.

## Current PR #12 status

The integration candidate has completed its planned merge gates:

- [x] full non-slow CPU suite;
- [x] Hydra basic sweep;
- [x] Optuna three-trial sweep;
- [x] cross-platform GitHub Actions;
- [x] full optional-dependency CI;
- [x] Code Quality PR;
- [x] CUDA `fast_dev_run` on a GTX 1660;
- [x] license/provenance review.

A full Docker image build/run remains useful as a post-merge smoke test, but it
is not a blocking merge criterion.

## 1. Full CPU environment

Create an isolated environment and install the target CPU PyTorch stack:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.14,<2.15" \
  "torchvision>=0.29,<0.30"

python -m pip install -r requirements/all.txt
python -m pip install -e .
```

Run the ordinary full-stack suite without long-running sweep tests:

```bash
python -m pytest -m "not slow" -v
```

Then run the integration smoke tests:

```bash
python -m pytest \
  tests/test_sweeps.py::test_example_experiment \
  tests/test_sweeps.py::test_hydra_sweep \
  tests/test_sweeps.py::test_optuna_sweep \
  -v
```

## 2. GitHub Actions on this fork

The Tests workflow supports manual dispatch and is enabled for this fork.

The workflow contains:

- cross-platform core tests on Linux, Windows, and macOS;
- a vision dependency job;
- a full optional-dependency job;
- selected Hydra/Optuna integration smoke tests;
- coverage collection.

PR #12 also runs the Code Quality PR workflow, which covers formatting,
docstring coverage, Flake8, Bandit, YAML/Markdown formatting, ShellCheck,
codespell, and nbQA.

## 3. CUDA smoke test

After CPU tests pass, verify the target machine's NVIDIA driver and run at least
the MNIST path on one GPU:

```bash
python src/train.py \
  trainer=gpu \
  ++trainer.fast_dev_run=true \
  logger=[]
```

PR #12 was validated on a GeForce GTX 1660 with PyTorch 2.14.0+cu130,
torchvision 0.29.0+cu130, and CUDA runtime 13.0. Lightning selected GPU 0 and
completed train, validation, and test with exit code 0.

For Docker:

```bash
docker build -t lightning-hydra:dev .

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/logs:/workspace/logs" \
  lightning-hydra:dev \
  python src/train.py trainer=gpu ++trainer.fast_dev_run=true logger=[]
```

## 4. Feature experiments

The shipped image experiments compose in CI without requiring user data.
End-to-end runs require manifests under the configured `data/` paths:

```bash
python src/train.py experiment=image_classification
python src/train.py experiment=image_multitask
python src/train.py experiment=image_reid
python src/train.py experiment=image_vicreg
```

See the corresponding files in `docs/` for manifest layout and task-specific
settings.

## Ready-to-merge criteria

- [x] full non-slow CPU suite passes;
- [x] Hydra basic and Optuna smoke tests pass;
- [x] at least one CUDA fast-dev run passes;
- [x] no unresolved license/provenance concern;
- [x] original ashleve MIT notice is preserved;
- [x] Tests and Code Quality PR workflows are green.
