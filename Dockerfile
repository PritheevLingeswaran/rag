# Production image (Render free tier). Multi-stage, non-root, no secrets.
#
# - Models are downloaded at BUILD time into the HF cache and baked into
#   the image: cold start never depends on (rate-limited) HF downloads.
# - Corpus ships in the image; the index is built at boot (~60 chunks,
#   measured ~53s total cold start at 0.1 CPU). At 10k+ chunks, switch to
#   pulling a prebuilt FaissStore version from object storage instead
#   (Stage 2 machinery exists; docs/infrastructure.md).
# - All secrets (API_KEYS, DATABASE_URL, REDIS_URL, GEMINI_API_KEY,
#   ALERT_WEBHOOK_URL) come from the platform's environment at runtime.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Bake the ONNX models into the image at build time.
ENV HF_HOME=/opt/hf-cache
RUN /opt/venv/bin/python -c "\
from huggingface_hub import hf_hub_download; \
[hf_hub_download(r, f) for r, f in [ \
  ('Xenova/all-MiniLM-L6-v2', 'onnx/model_quantized.onnx'), \
  ('Xenova/all-MiniLM-L6-v2', 'tokenizer.json'), \
  ('Xenova/ms-marco-MiniLM-L-6-v2', 'onnx/model_quantized.onnx'), \
  ('Xenova/ms-marco-MiniLM-L-6-v2', 'tokenizer.json')]]"

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home ragp

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache

WORKDIR /srv/ragp
COPY app/ app/
COPY data/ data/
COPY configs/ configs/
COPY migrations/ migrations/
COPY frontend/ frontend/
COPY PRIVACY.md PRIVACY.md

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    SERVE_PIPELINE=true

USER ragp
EXPOSE 8000

# uvicorn only starts accepting connections after the lifespan completes
# (models loaded, index built, pipeline warmed), so /health returning 200
# genuinely means READY -- suitable for the platform health check.
# --no-access-log: privacy policy commitment -- the app writes no
# per-request IP logs (the host's edge logs under its own policy).
# --proxy-headers: honor X-Forwarded-Proto from Render's edge so
# request.base_url is https (OAuth redirect URIs must match exactly).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log --proxy-headers --forwarded-allow-ips '*'"]
