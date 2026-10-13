# CPU image used by CI and the kind-based Argo e2e test.
# For GPU nodes build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG GIT_SHA=unknown
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=true PIP_NO_CACHE_DIR=1 \
    UXVML_CODE_VERSION=${GIT_SHA}

RUN useradd --create-home --uid 1000 uxv
WORKDIR /app

COPY pyproject.toml README.md ./
COPY uxvml ./uxvml
RUN pip install --index-url ${TORCH_INDEX} torch && pip install ".[s3]"
COPY configs ./configs

USER 1000
ENTRYPOINT ["uxvml"]
