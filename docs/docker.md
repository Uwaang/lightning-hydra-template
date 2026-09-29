# Docker

The Docker image is intended to provide a reproducible CUDA development and
training environment without copying the older CUDA 11.7 setup from the
reference project.

## Base image

The default base image is:

```text
pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime
```

Override it at build time if another supported PyTorch image is required:

```bash
docker build \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime \
  -t lightning-hydra:dev .
```

## Build

```bash
docker build -t lightning-hydra:dev .
```

The container runs as an unprivileged `app` user by default. On Linux, pass
your host UID/GID when you want bind-mounted files to retain host ownership:

```bash
docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -t lightning-hydra:dev .
```

## Train with an NVIDIA GPU

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit.

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/logs:/workspace/logs" \
  lightning-hydra:dev \
  python src/train.py trainer=gpu
```

Hydra overrides can be appended normally:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/logs:/workspace/logs" \
  lightning-hydra:dev \
  python src/train.py trainer=gpu trainer.max_epochs=2
```

## CPU smoke run

The CUDA-enabled image can also execute the CPU configuration:

```bash
docker run --rm lightning-hydra:dev \
  python src/train.py trainer=cpu trainer.fast_dev_run=true
```

Large training data, checkpoints, ONNX files, TensorRT engines, and local
environment files are excluded from the Docker build context by default.
