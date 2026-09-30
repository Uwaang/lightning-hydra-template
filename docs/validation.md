# Validation guide

Use this checklist before marking the full integration PR ready for review.

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

The Tests workflow supports manual dispatch.

If GitHub has workflows disabled for the fork:

1. open the repository's **Actions** tab;
2. enable workflows for the fork using GitHub's fork-workflow control;
3. open the **Tests** workflow;
4. choose **Run workflow**;
5. select `integration/full-stack`.

The workflow contains:

- cross-platform core tests on Linux, Windows, and macOS;
- a vision dependency job;
- a full optional-dependency job;
- selected Hydra/Optuna integration smoke tests;
- coverage collection.

## 3. CUDA smoke test

After CPU tests pass, verify the target machine's NVIDIA driver and container
runtime, then run at least the MNIST path on one GPU:

```bash
python src/train.py \
  trainer=gpu \
  ++trainer.fast_dev_run=true \
  logger=[]
```

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

Mark the integration PR ready only after:

- the full non-slow CPU suite passes;
- Hydra basic and Optuna smoke tests pass;
- at least one CUDA fast-dev run passes;
- no new license/provenance concern is introduced;
- the PR diff still preserves the original ashleve MIT notice.
