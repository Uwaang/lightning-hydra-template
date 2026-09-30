ARG PYTORCH_IMAGE=pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG USER_NAME=app
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG INSTALL_OPTIONAL=true

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        git-lfs \
        python3-venv \
        tini \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /workspace

COPY requirements.txt setup.py ./
COPY requirements ./requirements
RUN python -m pip install --upgrade pip \
    && python -m pip install -c requirements/docker-constraints.txt -r requirements.txt \
    && if [ "${INSTALL_OPTIONAL}" = "true" ]; then \
         python -m pip install -c requirements/docker-constraints.txt -r requirements/all.txt; \
       fi

RUN if ! getent group "${GROUP_ID}" >/dev/null; then \
        groupadd --gid "${GROUP_ID}" "${USER_NAME}"; \
    fi \
    && if ! getent passwd "${USER_ID}" >/dev/null; then \
        useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home "${USER_NAME}"; \
    fi \
    && mkdir -p "/home/${USER_NAME}" \
    && chown "${USER_ID}:${GROUP_ID}" "/home/${USER_NAME}"

ENV HOME="/home/${USER_NAME}"

COPY --chown=${USER_ID}:${GROUP_ID} . .
RUN python -m pip install -e .

RUN mkdir -p /workspace/data /workspace/logs \
    && chown -R "${USER_ID}:${GROUP_ID}" /workspace

USER ${USER_ID}:${GROUP_ID}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "src/train.py"]
