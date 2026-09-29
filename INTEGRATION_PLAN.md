# Lightning-Hydra integration plan

This fork uses `ashleve/lightning-hydra-template` as the modern baseline and
ports selected capabilities from
`gorodnitskiy/yet-another-lightning-hydra-template` instead of carrying its
PyTorch Lightning 1.8 runtime forward.

## Baseline

- Upstream baseline: `ashleve/lightning-hydra-template@bddbc24b82ab`
- Target runtime: Python 3.10+
- Target PyTorch: 2.14.x
- Target Lightning: 2.6.x
- Target Hydra: stable 1.3.x

## Migration phases

1. Modernize dependencies and CI while preserving the upstream MNIST smoke test.
2. Port reproducibility utilities, prediction export, and state-dict export.
3. Add a modern CUDA Docker workflow.
4. Port generic image datasets, Albumentations transforms, and HDF5 support.
5. Introduce model adapters for torchvision, timm, and segmentation-models-pytorch.
6. Port reusable losses and metrics.
7. Add single-task classification as the first non-MNIST end-to-end example.
8. Reimplement multi-dataset and multi-head training using Lightning 2.x APIs.
9. Port ReID/GeM and VICReg support.
10. Add optional Grad-CAM and experiment-tracking integrations.

## Migration rules

- Do not copy Lightning 1.x glue code unchanged.
- Keep `data` and `model` as the primary Hydra config groups.
- Keep optional CV dependencies out of the minimal core installation.
- Add tests with each ported capability before moving to the next phase.
- Preserve upstream license notices and audit licensing for newly ported code.
