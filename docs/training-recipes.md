# Training recipes and transfer learning

This repository keeps recipe features opt-in so the default training path remains simple.

## Label smoothing

The shipped classification losses expose PyTorch cross-entropy label smoothing:

```yaml
model:
  loss:
    label_smoothing: 0.1
```

The default is `0.0`. MixUp and CutMix already produce soft targets, so combining them with
additional label smoothing is a deliberate hyperparameter choice rather than an assumed default.

## Exponential moving average

Lightning 2.6 provides weight averaging directly. The `callbacks=ema` recipe configures
`WeightAveraging` with PyTorch's EMA averaging function and a default decay of `0.999`:

```bash
uv run train-command experiment=image_classification callbacks=ema
```

During validation Lightning evaluates the averaged weights, and the callback persists its
averaging state in checkpoints. The default recipe averages parameters and buffers. The feature
is opt-in because it adds a second model copy and therefore extra memory usage.

Use `callbacks.ema.avg_fn.decay=...` to tune the decay.

## Pretrained initialization

Torchvision and timm adapters already expose their native pretrained initialization controls:

```yaml
# torchvision
net:
  _target_: src.models.components.adapters.TorchvisionClassifier
  model_name: resnet18
  weights: DEFAULT

# timm
net:
  _target_: src.models.components.adapters.TimmClassifier
  model_name: resnet18
  pretrained: true
```

Pretrained preprocessing must still match the selected model. The generic image-classification
example uses the existing dataset transform configuration; model-specific automatic preprocessing
is a separate future capability.

## Head warmup and backbone fine-tuning

`TorchvisionClassifier` and `TimmClassifier` expose a small transfer-learning contract:

- `head_parameters()` returns the task-specific classifier head;
- `backbone_parameters()` returns the remaining model parameters;
- `set_backbone_trainable(bool)` freezes or unfreezes only the backbone.

The `callbacks=finetune` recipe uses `BackboneUnfreezingCallback`. It freezes the backbone when
fit starts and restores backbone gradients at `unfreeze_at_epoch`:

```bash
uv run train-command experiment=image_classification \
  model.net.weights=DEFAULT \
  callbacks=finetune \
  callbacks.backbone_unfreezing.unfreeze_at_epoch=2
```

`callbacks=finetune_ema` combines staged backbone unfreezing with EMA.

The callback does not create a separate optimizer parameter group. Optimizers in the template are
constructed from all parameters; frozen backbone tensors simply have no gradients until they are
unfrozen. After unfreezing they use the optimizer's current learning rate. Differential backbone
learning rates are intentionally left for a later recipe rather than hidden inside the callback.

## Reference fine-tuning experiment

`experiment=image_classification_finetune` demonstrates a common starting point:

- torchvision `DEFAULT` pretrained weights;
- one epoch of classifier-head warmup;
- full-backbone fine-tuning afterwards;
- label smoothing `0.1`;
- EMA with decay `0.999`.

The values are examples, not universal best settings. They are suitable candidates for later HPO
rather than hard-coded recommendations.

## Limitations

- The staged fine-tuning callback currently targets the repository's torchvision/timm classifier
  adapters; arbitrary custom networks must implement `set_backbone_trainable(bool)` to opt in.
- Differential learning rates and gradual layer-wise unfreezing are not included yet.
- EMA/weight averaging is not intended for sharded-model setups without separate validation.
- Real pretrained runs may download weights and therefore are not executed in ordinary offline CI.
