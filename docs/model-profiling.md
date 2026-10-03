# Model complexity and Pareto analysis

The repository provides dependency-free utilities for comparing model cost alongside training
and runtime metrics.

## Complexity report

`profile_model_complexity()` runs one representative inference pass and reports:

- total parameters;
- trainable parameters;
- serialized `state_dict` size;
- MACs for `Conv1d`, `Conv2d`, `Conv3d`, and `Linear` modules;
- FLOPs using the explicit convention `FLOPs = 2 * MACs`;
- peak CUDA allocated memory above the pre-forward baseline when CUDA inputs are used.

```python
import torch

from src.utils import profile_model_complexity, save_model_complexity_report

model = ...
example = (torch.randn(1, 3, 224, 224),)

report = profile_model_complexity(model, example)
save_model_complexity_report(report, "artifacts/model-complexity.json")
```

The profiler restores the model's original train/eval state after the representative forward.

### MAC/FLOP scope

The built-in counter deliberately avoids pretending to understand every custom operator. Its
`macs` field counts standard convolution and linear modules only. Functional matrix
multiplication, custom CUDA kernels, attention implementations, and fused operators can therefore
make the reported MAC/FLOP total a lower bound. The JSON report records the counted operator
families so downstream comparisons can keep the convention explicit.

For model families that depend heavily on unsupported operators, use a task-specific profiler or
add a validated counter rather than silently mixing conventions.

CPU peak native tensor memory is also intentionally omitted: process RSS deltas are noisy and do
not provide a reliable inference-peak contract. CUDA peak allocated memory is available when a
real CUDA run is performed.

## Pareto frontier

`pareto_front()` accepts arbitrary records plus explicit minimization/maximization objectives.

```python
from src.utils import ParetoObjective, pareto_front

records = [
    {"name": "small", "accuracy": 0.87, "params": 2_000_000, "macs": 100_000_000},
    {"name": "large", "accuracy": 0.91, "params": 5_000_000, "macs": 240_000_000},
    {"name": "dominated", "accuracy": 0.85, "params": 6_000_000, "macs": 300_000_000},
]

frontier = pareto_front(
    records,
    [
        ParetoObjective("accuracy", "max"),
        ParetoObjective("params", "min"),
        ParetoObjective("macs", "min"),
    ],
)
```

The result preserves input order and returns only nondominated records. This utility is
independent of Optuna; a later multi-objective sweep can feed its trial results into the same
Pareto representation.

## Recommended comparison contract

For reproducible model-family comparisons, keep the following fixed across candidates:

- identical representative input shape;
- identical MAC/FLOP convention;
- identical dataset split and evaluation metric;
- identical runtime benchmark batch/thread/provider settings;
- actual target hardware when latency or memory is part of the decision.

This keeps literature FLOP/MAC convention differences from leaking into the repository's own
Pareto comparisons.
