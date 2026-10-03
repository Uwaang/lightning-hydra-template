# Training recipes and fine-tuning

This repository keeps reusable training policies opt-in. The default classification path remains
plain supervised training unless a recipe is explicitly selected.

## Label smoothing

Classification configs expose PyTorch cross-entropy label smoothing directly:

```bash
uv run train-command experiment=image_classification model.loss.label_smoothing=0.1
```

The default is `0.0`. MixUp/CutMix can still provide soft targets independently; avoid assuming
that stronger smoothing is automatically better when batch mixing is already enabled.

## Exponential moving average

Lightning 2.6 provides `WeightAveraging`, so EMA uses the framework callback rather than a custom
shadow-model implementation. The preset uses PyTorch `get_ema_multi_avg_fn` with decay `0.999`:

```bash
uv run train-command experiment=image_classification callbacks=ema
```

Lightning swaps averaged weights in for validation and persists the averaging state in
checkpoints. Override the decay through
`callbacks.ema.multi_avg_fn.decay=<value>` when an experiment needs it.

## Public pretrained weights

Torchvision adapters already accept string weight names, including `DEFAULT`. A convenience model
preset enables the default pretrained weights:

```bash
uv run train-command \
  experiment=image_classification \
  model=image_classification_pretrained
```

timm adapters expose the equivalent `pretrained=true` flag. Pretrained weights do not remove the
need to align input normalization/resolution with the selected weight recipe.

## Loading project checkpoints for transfer learning

`ClassificationLitModule` can initialize its network from either a plain state dict or a Lightning
checkpoint with safe `weights_only=True` deserialization:

```bash
uv run train-command \
  experiment=image_classification \
  model.pretrained_checkpoint=/path/to/source.ckpt \
  model.pretrained_strip_prefix=net. \
  model.pretrained_ignore_shape_mismatch=true
```

`pretrained_ignore_shape_mismatch=true` is useful when the source classifier head and destination
head have different shapes. Matching backbone tensors load while incompatible head tensors are
skipped. Keep it disabled when an exact architecture match is required.

## Head warm-up then full fine-tuning

The `finetuning` callback freezes all parameters except configured `fnmatch` name patterns before
optimizer creation. At the selected epoch it unfreezes the remaining parameters and adds them as a
new optimizer parameter group with a lower initial learning rate.

```bash
uv run train-command \
  experiment=image_classification \
  model=image_classification_pretrained \
  callbacks=finetuning \
  trainer.max_epochs=10 \
  callbacks.finetuning.unfreeze_at_epoch=2
```

The shipped pattern list covers common torchvision/timm classifier names:

- `net.model.fc.*`
- `net.model.classifier.*`
- `net.model.head.*`
- `net.model.heads.*`

For a custom model, override `callbacks.finetuning.trainable_patterns`. The callback fails early if
no parameter matches, rather than silently training nothing.

## Sample-level augmentation policy presets

The default sample pipeline remains basic resize/flip. Three torchvision v2 presets are available
without adding another augmentation library:

```bash
uv run train-command experiment=image_classification data=image_classification_randaugment
uv run train-command experiment=image_classification data=image_classification_trivialaugment
uv run train-command experiment=image_classification data=image_classification_augmix
```

The policy is inserted after geometric sample operations and before float conversion/normalization.
Batch-level MixUp/CutMix remains a separate data-loader policy.
