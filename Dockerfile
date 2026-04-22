# Two-stage Dockerfile: build the React bundle, then assemble a Python
# image that serves the API + the built SPA from one process.
#
#   docker build -t text-autocomplete .
#   docker run --rm -p 8010:8010 text-autocomplete
#
# Visit http://localhost:8010 for the UI and /docs for Swagger. For the
# optional neural paths, build with `--build-arg INSTALL_TORCH=1` to
# include torch + transformers + safetensors (~3 GB image). CPU-only by
# default; see docker-compose.yml for a GPU-enabled variant.

# -----------------------------------------------------------------------------
# Stage 1 — build the React SPA
# -----------------------------------------------------------------------------
FROM node:24-alpine AS frontend-build
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps

COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2 — Python runtime with FastAPI + optional torch
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ARG INSTALL_TORCH=0
WORKDIR /app

# System deps kept minimal. curl is handy for container healthchecks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install --no-cache-dir \
            torch safetensors transformers \
            prometheus-fastapi-instrumentator redis ; \
    else \
        pip install --no-cache-dir \
            prometheus-fastapi-instrumentator redis ; \
    fi

COPY src/ ./src/
COPY cli.py ./

# Pull in the built SPA from stage 1 so the StaticFiles mount finds it.
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8010/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8010"]
