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
- Optuna sweeping uses a separate modern plugin environment instead of the legacy stable 1.2.0 stack.
- VICReg supports unlabeled manifests and monitors `val/loss`.
- Multi-head prediction export recursively handles nested batched mappings.
- HDF5 file handles are opened lazily per process.
- Image experiments are config-composed in ordinary CI without requiring user datasets.
- Runtime sweep smoke tests are separated from the cross-platform non-slow suite.
- README and feature-parity documentation describe the integrated extensions.

## Validation completed

Dependency-free validation on an isolated clone:

- 59 Python source/test/script files parse successfully.
- 45 local Hydra targets resolve to existing symbols.
- Optional requirement include paths resolve.
- No optional third-party package is imported at top level from `src/`.
- No unresolved merge markers.
- `git diff --check` passes.
- Shell scripts pass `bash -n`.
- Every non-deleted file contributed by the ten modular feature branches is present in the full-stack tree.

Individual implementation paths were also exercised during development.
A separate isolated sandbox with PyTorch 2.10 CPU, torchvision 0.25,
Lightning 2.6.5, and TorchMetrics 1.9 successfully ran:

- a real Lightning `Trainer.fit(fast_dev_run=True)` with the integrated
  multi-task `CombinedLoader(max_size_cycle)` pattern;
- angular-margin and VICReg backward passes;
- ResNet-18, MobileNetV3-Small, and ViT-B/16 classifier-head replacement;
- ResNet-18 `layer4 -> GeM -> projection -> L2 normalization` for ReID;
- nested HDF5 image-byte roundtrip; and
- per-sample splitting of nested multi-head prediction outputs.

The exact target PyTorch 2.14 / torchvision 0.29 environment still requires
the full dependency-install test gate below.

## Remaining merge gates

1. **Full dependency pytest:** pending. GitHub Actions currently creates no runs
   for this fork, and the connected local execution bridge requires separate
   approval before it can install network packages.
2. **GPU smoke test:** run at least one CUDA `fast_dev_run` after the CPU/full
   suite is green.
3. **License/provenance review:** preserve the upstream MIT notice and confirm
   that any reference-derived implementation is safe to publish before merging
   the integration PR.

## Migration rules

- Do not copy Lightning 1.x glue code unchanged.
- Keep `data` and `model` as the primary Hydra config groups.
- Keep optional CV dependencies out of the minimal core installation.
- Add tests with each integrated capability.
- Prefer maintained provider/library APIs over model-structure heuristics.
- Preserve upstream license notices and document provenance for integrated work.
