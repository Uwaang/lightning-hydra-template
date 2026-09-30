# Reference feature parity

This project starts from `ashleve/lightning-hydra-template` and selectively
ports the useful behavior of
`gorodnitskiy/yet-another-lightning-hydra-template`.

## Ported or reimplemented

- modern Lightning/Hydra project skeleton
- image manifests and unlabeled prediction datasets
- Albumentations transforms
- HDF5-backed image storage
- multiple prediction dataloaders
- explicit multi-task training with CombinedLoader
- torchvision / timm / segmentation-models-pytorch adapters
- multi-head classification
- Focal Loss
- ArcFace / SphereFace / CosFace
- VICReg
- ReID embeddings and GeM
- prediction JSON/CSV export
- plain state-dict export
- run metadata snapshots
- CUDA Docker environment
- Grad-CAM
- Git LFS rules for large generated artifacts

## Replaced by existing baseline features

- custom terminal progress bar -> Lightning RichProgressBar
- custom W&B checkpoint callback -> WandbLogger `log_model`
- manual random seeding -> Lightning `seed_everything(..., workers=True)`
- custom metric copies -> maintained TorchMetrics implementations
- train/eval shell wrappers -> Python entry points and Makefile targets

## Intentionally not ported

- TensorFlow as a core dependency solely to read TensorBoard event files
- `shell=True` metadata collection
- Git LFS for every JSON file
- Lightning 1.x epoch-end hooks
- positional assumptions such as "last model child is the classifier"
- the reference environment's incorrect `torch.cuda.deterministic` /
  `torch.cuda.benchmark` assignments

Optional TensorBoard event analysis can be added later with a lightweight
event reader if a real use case requires it; pulling TensorFlow into the core
template for that purpose is deliberately avoided.
