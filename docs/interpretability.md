# Interpretability

Grad-CAM support is optional and uses the current `grad-cam` package API.

## Install

```bash
uv sync --extra interpretability
```

## Example

For the template's torchvision classifier wrapper, the underlying ResNet lives
under `model`, so a typical target layer is:

```python
from src.utils.gradcam import compute_gradcam

masks = compute_gradcam(
    model=classifier,
    input_tensor=batch,
    target_layer="model.layer4.1",
)
```

If `categories=None`, pytorch-grad-cam targets the highest-scoring class for
each sample. Pass one category integer per batch element to explain explicit
classes.

`overlay_gradcam` accepts RGB HWC float images in `[0, 1]` and returns
uint8 RGB overlays.

## Why this differs from the reference repository

The reference helper directly embeds assumptions about batch keys, sigmoid
classification, plotting layout, and the old pytorch-grad-cam constructor.
This wrapper keeps model explanation independent of the dataset and plotting
surface, and resolves target layers explicitly by module path.
