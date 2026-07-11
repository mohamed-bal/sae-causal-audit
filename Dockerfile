FROM python:3.12-slim@sha256:47b3c77bb7f66b4c81a09d0c0f7d3f2b8e4a0e10 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MKL_CBWR=COMPATIBLE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=0

WORKDIR /work

RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
RUN pip install -e ".[dev]"

CMD ["make", "reproduce"]
