# Multi-task image classification

The legacy reference template returned a dictionary of train dataloaders and
relied on Lightning 1.x multiple-loader behavior. This implementation makes
the policy explicit with Lightning 2.x `CombinedLoader`.

## Training loader policy

`MultiTaskImageDataModule.train_mode` supports:

- `min_size`
- `max_size_cycle`
- `max_size`

The default is `max_size_cycle`, matching the practical behavior of the older
template while making the choice visible in config.

Validation and test dataloaders are returned in the same deterministic order
as the named training tasks. `MultiHeadClassificationLitModule` maps
`dataloader_idx` back to that task order.

## Shared backbone

`MultiHeadClassifier` runs one shared feature extractor and maintains a
separate linear head per named task. During training, each task may receive a
different image batch from the CombinedLoader.

## Run

```bash
python src/train.py experiment=image_multitask
```

Edit the `heads` mapping and the matching dataset names together. The example
uses `label_a` and `label_b`.
