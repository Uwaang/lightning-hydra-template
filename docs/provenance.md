# Provenance and licensing notes

This repository is an integration fork built from two public reference
projects with different licensing clarity.

## Primary baseline

The codebase starts from
[`ashleve/lightning-hydra-template`](https://github.com/ashleve/lightning-hydra-template).

The upstream README explicitly states that Lightning-Hydra-Template is
licensed under the MIT License and includes the original notice:

> Copyright (c) 2021 ashleve

That notice remains present in this repository's README and should be
preserved when redistributing substantial portions of the original template.

## Secondary behavioral reference

[`gorodnitskiy/yet-another-lightning-hydra-template`](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
was used as a behavioral and feature reference for image data handling,
multi-task training, ReID/GeM, VICReg, export utilities, and interpretability.

At the time of the 2026 integration review, its `main` tree contained no
`LICENSE`, `LICENCE`, `COPYING`, or `NOTICE` file, and its README did not
state a software license. Therefore its source code should not be treated as
licensed for direct copying merely because the repository is public.

## Integration policy

The secondary repository's useful capabilities were independently
reimplemented against current public APIs rather than vendored as source
files:

- Lightning 2.x `CombinedLoader` instead of Lightning 1.x multiple-loader glue
- provider-specific torchvision/timm/SMP adapters instead of child-order model introspection
- torchvision feature extraction for ReID instead of positional layer replacement
- modern pytorch-grad-cam API instead of the reference helper
- process-aware HDF5 readers and pathlib-based path handling
- vectorized angular-margin objectives and maintained TorchMetrics metrics
- safe subprocess metadata collection instead of `shell=True`

An exact stripped-line comparison of the major corresponding implementation
files found low overlap overall. The highest overlap was in short mathematical
loss modules, where the matching lines were generic Python/PyTorch constructs
such as imports, constructor signatures, `super().__init__()`, and standard
normalization operations.

## Publication gate

Before publishing or distributing this fork in a context where licensing
provenance is important:

1. preserve the original ashleve MIT notice;
2. do not replace integrated implementations with copied code from the
   unlicensed secondary reference unless its author later supplies compatible
   licensing terms;
3. review any future copied snippets or patches individually for their source
   license and attribution requirements.

This document records engineering provenance and is not legal advice.
