# Lightning-Hydra integration status

This fork uses `ashleve/lightning-hydra-template` as the modern baseline and
reimplements selected capabilities from
`gorodnitskiy/yet-another-lightning-hydra-template` instead of carrying its
PyTorch Lightning 1.8 runtime forward.

## Baseline

- Upstream baseline: `ashleve/lightning-hydra-template@bddbc24b82ab`
- Runtime target: Python 3.10+
- PyTorch target: 2.14.x
- Lightning target: 2.6.x
- Hydra target: stable 1.3.x
- Full integration branch: `integration/full-stack`
- Full integration PR: #12

## Migration status

1. ✅ Modernize dependencies and CI while preserving the upstream MNIST path.
2. ✅ Port reproducibility utilities, prediction export, and state-dict export.
3. ✅ Add a modern CUDA Docker workflow.
4. ✅ Port generic image datasets, Albumentations transforms, and HDF5 support.
5. ✅ Introduce provider-specific adapters for torchvision, timm, and segmentation-models-pytorch.
6. ✅ Port reusable losses and use TorchMetrics for maintained standard metrics.
7. ✅ Add generic single-task image classification.
8. ✅ Reimplement multi-dataset and multi-head training with Lightning 2.x `CombinedLoader`.
9. ✅ Add ReID/GeM and VICReg task paths.
10. ✅ Add optional Grad-CAM tooling and scoped large-artifact Git LFS rules.

## Integration hardening completed

- Optional vision/model/interpretability/sweep dependencies are separated from core.
- Albumentations and h5py are lazy imports rather than core import-time requirements.
- Optuna sweeping uses the tested `hydra-optuna-sweeper==1.2.0` +
  `optuna==2.10.1` pair with the stable Hydra 1.3.7 baseline.
- VICReg supports unlabeled manifests and monitors `val/loss`.
- Multi-head prediction export recursively handles nested batched mappings.
- HDF5 file handles are opened lazily per process.
- Image experiments are config-composed in ordinary CI without requiring user datasets.
- Runtime sweep smoke tests are separated from the cross-platform non-slow suite.
- Windows non-UTF consoles no longer crash Lightning Rich progress output.
- Pre-commit, Bandit, docstring coverage, nbQA, and shell line-ending handling are CI-clean.
- README, validation, feature-parity, and provenance documentation describe the integrated extensions.

## Validation completed

Static and dependency-free validation:

- 59 Python source/test/script files parse successfully.
- 45 local Hydra targets resolve to existing symbols.
- Optional requirement include paths resolve.
- No optional third-party package is imported at top level from `src/`.
- No unresolved merge markers.
- `git diff --check` passes.
- Shell scripts pass `bash -n`.
- Every non-deleted file contributed by the ten modular feature branches is present.

Exact target CPU validation:

- Python 3.12 + PyTorch 2.14.0 + torchvision 0.29.0 + Lightning 2.6.6.
- Local non-slow suite: 43 passed, 1 skipped, 10 deselected, 0 failed.
- MNIST `fast_dev_run`: passed.
- Hydra basic two-trial sweep: passed.
- Optuna three-trial sweep: passed.
- `pip check`: no broken requirements.

GitHub Actions on PR #12:

- Linux Python 3.10: passed.
- Linux Python 3.11: passed.
- Linux Python 3.12: passed.
- Windows Python 3.12: passed.
- macOS Python 3.12: passed.
- vision-tests: passed.
- full-stack-tests, including selected Hydra/Optuna smoke tests: passed.
- code coverage job: passed.
- Code Quality PR: passed.

Exact target GPU smoke:

- NVIDIA GeForce GTX 1660, driver 591.86.
- PyTorch 2.14.0+cu130 and torchvision 0.29.0+cu130.
- CUDA runtime 13.0; `torch.cuda.is_available() == True`.
- Lightning selected GPU 0 and completed MNIST train, validation, and test under
  `fast_dev_run` with exit code 0.

Additional component-level runtime checks covered `CombinedLoader`,
angular-margin/VICReg backward passes, torchvision head replacement, ReID/GeM,
nested HDF5 roundtrips, nested prediction serialization, and Grad-CAM module
resolution.

## Merge gates

All planned merge gates are complete:

1. ✅ Full dependency CPU pytest / integration smoke.
2. ✅ CUDA `fast_dev_run` on a real NVIDIA GPU.
3. ✅ License/provenance review with upstream MIT notice preserved and the
   unlicensed secondary reference treated as a behavioral reference rather than
   vendored source.

The Dockerfile and dependency graph are checked by repository validation, but a
full Docker image build/run was not used as a merge gate and remains an optional
post-merge smoke test.

## Migration rules

- Do not copy Lightning 1.x glue code unchanged.
- Keep `data` and `model` as the primary Hydra config groups.
- Keep optional CV dependencies out of the minimal core installation.
- Add tests with each integrated capability.
- Prefer maintained provider/library APIs over model-structure heuristics.
- Preserve upstream license notices and document provenance for integrated work.
