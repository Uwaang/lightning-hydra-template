# Imagewoof-160 timm recipe benchmark

This benchmark is intentionally small enough for a 6 GB consumer GPU while still exercising
the template's transfer-learning, fine-tuning, EMA, label-smoothing, and KD paths.

## Fixed protocol

- dataset: fast.ai Imagewoof-160 (`imagewoof2-160`)
- classes: 10 dog breeds
- input: 160 x 160
- student: `mobilenetv3_small_100.lamb_in1k`
- KD teacher: `resnet18.a1_in1k`
- batch size: 64
- precision: `16-mixed`
- epochs: 12
- seed: 12345
- optimizer: AdamW, weight decay 0.01
- schedule: cosine annealing, `eta_min=1e-6`
- augmentation: random resized crop + horizontal flip
- validation/test: resize to 176, center crop to 160
- MixUp/CutMix: disabled to isolate the recipe variables

Scratch uses learning rate `1e-3`; pretrained recipes use `3e-4`.

## Dataset split

The official Imagewoof train split is preserved. The official validation split is divided
deterministically and per class into equal validation and test subsets. This avoids selecting
hyperparameters and reporting the final score on exactly the same samples.

After extracting Imagewoof-160 under `data/imagewoof2-160`:

```bash
python scripts/prepare_imagewoof160.py data/imagewoof2-160 --seed 12345
```

The command creates `train.json`, `val.json`, `test.json`, `predict.json`, and `classes.json`
under `data/imagewoof2-160/manifests`.

## Runs

1. `imagewoof_timm_scratch`
2. `imagewoof_timm_pretrained`
3. `imagewoof_timm_freeze_unfreeze`
4. `imagewoof_timm_ema`
5. `imagewoof_timm_label_smoothing`
6. `imagewoof_timm_kd`
7. `imagewoof_timm_kd_label_smoothing` combines the two individually useful recipes.
8. `imagewoof_timm_teacher` is the ResNet-18 teacher-production run.

The freeze/unfreeze recipe trains only the new classifier head for epochs 0-1 and unfreezes
the backbone at epoch 2. EMA uses decay 0.999. Label smoothing uses 0.1. KD uses alpha 0.5 and
temperature 4.0.

The teacher run writes its best plain network state dict to:

```text
logs/benchmarks/imagewoof/teacher/best_state_dict.pt
```

The KD config consumes that path directly.

## Interpretation

Imagewoof is itself a subset of ImageNet. Therefore ImageNet-1k pretrained weights have already
seen the source classes. Treat this as an engineering benchmark of the template and its training
recipes, not as a leakage-free measurement of transfer to unseen semantic classes.

For a stronger transfer-learning claim, repeat the winning recipes on a target dataset outside
ImageNet, such as Oxford-IIIT Pet.

## Primary comparison

Compare `test/acc`, `val/acc_best`, wall-clock training time, peak GPU memory, parameter count,
and MACs/FLOPs. Run one seed first. Only rerun the closest top recipes with multiple seeds if the
single-seed result is ambiguous.
## First GTX 1660 result (seed 12345)

The primary top-1 column is ordinary sample accuracy from the generated classification report.
`test/acc` is also shown because the Lightning metric is TorchMetrics `MulticlassAccuracy`
with its default macro averaging.

| Recipe | Top-1 test | Macro test | Best macro val | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Scratch | 40.93% | 40.92% | 43.75% | 236.8 s |
| Pretrained full fine-tune | 84.39% | 83.58% | 83.81% | 206.8 s |
| Freeze 2 epochs then unfreeze | 84.34% | 83.60% | 83.78% | 208.7 s |
| Pretrained + EMA | 83.68% | 82.95% | 82.89% | 253.9 s |
| Pretrained + label smoothing 0.1 | 85.31% | 84.57% | 84.81% | 219.4 s |
| Pretrained + KD | **87.19%** | **86.41%** | **87.76%** | 419.4 s |
| Pretrained + KD + label smoothing | **87.19%** | 86.38% | 87.04% | 417.7 s |
| ResNet-18 teacher | 89.83% | 89.04% | 88.88% | 897.6 s |

Relative to pretrained full fine-tuning, KD improved ordinary top-1 by 2.80 percentage
points and macro accuracy by 2.83 points. Label smoothing improved ordinary top-1 by
0.92 points and macro accuracy by 0.99 points. Freeze/unfreeze was effectively neutral,
and EMA was slightly worse in this short 12-epoch setting. Combining label smoothing
with KD did not improve ordinary top-1 beyond KD alone and slightly reduced macro accuracy.

These are single-seed engineering results, not statistically robust model rankings.

At 160 x 160, the profiler reports 1,528,106 parameters and 28.83M counted MACs for the
MobileNetV3-small student, versus 11,181,642 parameters and 925.29M counted MACs for the
ResNet-18 teacher. The profiler counts Conv/Linear MACs and therefore should be interpreted
using its documented coverage rather than as a complete operator-level FLOP benchmark.

MLflow experiment: `imagewoof-timm-benchmark`.
