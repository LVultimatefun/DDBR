# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# Stage 1: Build Dual-Role Web Studio SPA (React 19 + TypeScript + Vite)
# -----------------------------------------------------------------------------
FROM node:22-slim AS web-builder

WORKDIR /build/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Runtime Python 3.12 + FFmpeg + FastAPI Studio & CLI
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROBO_VLM_QC_WEB_DIST=/app/web/dist

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY robo_vlm_qc/ ./robo_vlm_qc/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir .

COPY --from=web-builder /build/web/dist /app/web/dist

RUN mkdir -p /workspace /data

EXPOSE 8000

CMD ["robo-vlm-qc", "-w", "/workspace", "serve", "--host", "0.0.0.0", "--port", "8000"]
