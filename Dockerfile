# syntax=docker/dockerfile:1

########################################
# Stage 1: build a venv with uv
########################################
FROM python:3.11-slim AS builder

# build-essential: covers any dependency that doesn't ship a prebuilt wheel
# for this platform. Discarded with this stage, so it never reaches runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /build

# requirements.txt is the verified, working dependency set for this project,
# pinned against the actual Python 3.11 runtime it was tested with.
# pyproject.toml/uv.lock are out of sync (see README "Known Issues") and are
# intentionally not used here.
COPY requirements.txt .

RUN uv venv /opt/venv --python 3.11 \
    && uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt

########################################
# Stage 2: slim runtime image
########################################
FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime needed by torch/faiss-cpu at import time.
# curl: used by HEALTHCHECK below to probe Streamlit's health endpoint.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=appuser:appuser app.py main.py streamlit_app.py ./
COPY --chown=appuser:appuser src/ ./src/

# Mount points for volumes defined in docker-compose.yml; created up front so
# permissions are correct even before a volume is attached (e.g. plain `docker run`).
RUN mkdir -p /app/faiss_store /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
