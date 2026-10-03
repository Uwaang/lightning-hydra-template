# Hyperparameter optimization and Pareto search

Hydra remains the experiment-composition layer and the optional Hydra Optuna Sweeper provides HPO.
The repository supports both single-objective and ordered multi-objective returns.

## Single-objective HPO

The existing MNIST preset remains available:

```bash
uv run train-command -m hparams_search=mnist_optuna experiment=example
```

For generic image classification, `image_classification_optuna` provides a reusable starting
search space over:

- learning rate;
- weight decay;
- label smoothing;
- batch size;
- MixUp/CutMix mode and alpha values.

```bash
uv run train-command -m \
  experiment=image_classification \
  hparams_search=image_classification_optuna
```

The ranges are examples, not claims that one recipe is universally optimal.

## Multi-objective optimization

`optimized_metric` remains backward-compatible for one objective. Multi-objective runs instead set
`optimized_metrics` to an ordered list. That order must match `hydra.sweeper.direction`.

The built-in Pareto smoke preset optimizes:

```text
val/acc_best  -> maximize
model/params  -> minimize
```

with Optuna NSGA-II:

```bash
uv run train-command -m \
  hparams_search=mnist_optuna_pareto \
  experiment=example
```

`model/params` is added to every training run's returned metric dictionary. When a Lightning
module exposes `net`, the metric counts that task network rather than auxiliary training-only
modules such as a KD teacher.

The Hydra entrypoint returns an ordered tuple for multi-objective runs. CI executes a real
two-objective Optuna smoke study rather than only checking config composition.

## Relation to model profiling

The separate model-profiling utilities can compute Params, MACs/FLOPs, serialized size, CUDA peak
memory and Pareto fronts. Params are safe to expose automatically because they do not need a
representative input.

MACs/FLOPs are intentionally not injected into every training run yet: they require a defined
input signature and the built-in counter has an explicit supported-op scope. Latency is even more
target-dependent. For those objectives, profile under a fixed input/hardware contract and merge
the resulting records into `pareto_front()` rather than silently comparing mismatched conventions.

## Practical progression

A useful workflow is:

```text
coarse single-objective HPO
  -> narrow promising ranges
  -> multi-objective accuracy/params search
  -> profile finalists for MACs and target latency
  -> compute final Pareto frontier
```

This keeps expensive hardware measurements out of early HPO while still preserving the final
deployment trade-offs.
