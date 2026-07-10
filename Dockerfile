# Fully-pinned reproduction environment for the toy-regime audit.
# The "every number is regenerable" claim becomes: docker build + make reproduce.
FROM python:3.12-slim@sha256:47b3c77bb7f66b4c81a09d0c0f7d3f2b8e4a0e10 AS base
# NOTE: replace the digest above with the current python:3.12-slim digest at
# first build (docker pull python:3.12-slim && docker inspect); a stale digest
# fails loudly, which is the desired behavior for a pinned scientific image.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MKL_CBWR=COMPATIBLE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /work

# CPU-only torch keeps the image small; the toy regime needs no GPU.
# Pinned to match .github/workflows/ci.yml exactly — the whole point of a
# pinned Docker image is that "the pinned environment" means the same thing
# everywhere it's invoked.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
RUN pip install -e ".[dev]"

# Default: run the full test suite, then regenerate all toy-regime results.
CMD ["make", "reproduce"]
