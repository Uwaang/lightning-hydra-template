ARG PYTORCH_IMAGE=pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG USER_NAME=app
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG INSTALL_OPTIONAL=true

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        git-lfs \
        tini \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# The official PyTorch CUDA image owns torch/torchvision. The project lockfile
# intentionally resolves CPU wheels for local development and CI, so exporting
# those two packages into this image would replace the CUDA-enabled runtime.
COPY pyproject.toml uv.lock ./
RUN python - <<'PY'
import torch
import torchvision

assert torch.__version__.startswith("2.14."), torch.__version__
assert torchvision.__version__.startswith("0.29."), torchvision.__version__
assert torch.cuda._is_compiled(), "Base image must contain a CUDA-enabled PyTorch build."
print(f"base torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}")
PY

RUN if [ "${INSTALL_OPTIONAL}" = "true" ]; then \
      EXTRA_ARGS="--extra all"; \
    else \
      EXTRA_ARGS=""; \
    fi \
    && /bin/uv export --frozen ${EXTRA_ARGS} \
         --no-emit-project \
         --no-emit-package torch \
         --no-emit-package torchvision \
         --format requirements.txt \
         --output-file /tmp/locked-requirements.txt \
    && /bin/uv pip install --system --break-system-packages --no-deps --requirement /tmp/locked-requirements.txt \
    && rm -f /tmp/locked-requirements.txt

RUN if ! getent group "${GROUP_ID}" >/dev/null; then \
      groupadd --gid "${GROUP_ID}" "${USER_NAME}"; \
    fi \
    && if ! getent passwd "${USER_ID}" >/dev/null; then \
      useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --home-dir "/home/${USER_NAME}" "${USER_NAME}"; \
    fi \
    && mkdir -p "/home/${USER_NAME}"

COPY --chown=${USER_ID}:${GROUP_ID} . .
RUN /bin/uv pip install --system --break-system-packages --no-deps --editable .

RUN mkdir -p /workspace/data /workspace/logs "/home/${USER_NAME}" \
    && chown -R "${USER_ID}:${GROUP_ID}" /workspace "/home/${USER_NAME}"

ENV HOME=/home/${USER_NAME}
USER ${USER_ID}:${GROUP_ID}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "src/train.py"]
