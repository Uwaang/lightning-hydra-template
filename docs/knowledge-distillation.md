# Knowledge distillation

The generic classification module supports optional logit knowledge distillation without changing
the default student-only training path.

## Training semantics

For student logits `z_s`, teacher logits `z_t`, target `y`, temperature `T`, and distillation
weight `alpha`, the training loss is:

```text
(1 - alpha) * supervised_loss(z_s, y)
+ alpha * T^2 * KL(softmax(z_t / T) || softmax(z_s / T))
```

The teacher:

- is registered as a module so Lightning moves it to the correct device;
- is forced to evaluation mode even while the student trains;
- has `requires_grad=False` for every parameter;
- is excluded from the optimizer;
- must emit logits with exactly the same shape as the student.

Distillation is applied only during `training_step`. Validation and test loss remain the normal
supervised criterion on the student, keeping model-selection metrics independent of teacher loss.

## Teacher checkpoints

The repository's plain network state-dict exports are the recommended teacher artifact. A teacher
training run can produce `best_state_dict.pt`, and the KD run loads that file directly into the
configured teacher network.

```bash
uv run train-command \
  experiment=image_classification \
  model=image_classification_kd \
  model.teacher_state_dict_path=/path/to/teacher/best_state_dict.pt
```

The shipped KD recipe uses a torchvision ResNet-18 student and ResNet-50 teacher only as an
example. Override either model as needed. The teacher checkpoint path is mandatory in the recipe
so a random teacher cannot be used accidentally.

Plain state-dict export from a KD run automatically excludes the `teacher.` prefix. Lightning
checkpoints still retain teacher state because it is required for exact training resume.

## Main controls

```yaml
distillation_alpha: 0.5
distillation_temperature: 4.0
```

`distillation_alpha=0` disables the KD term. Positive alpha requires a teacher. Temperature must
be positive.

MixUp/CutMix remains compatible: the supervised term can use its soft target while the teacher
provides logits for the same mixed image.

## Scope

The current capability deliberately covers standard logits-only distillation. It does not yet
implement feature-map matching, attention transfer, relational KD, or teacher ensembles. Those
should be added only when an experiment requires them rather than expanding the core abstraction
preemptively.
