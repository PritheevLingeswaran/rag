# Image for running the load test / memory gate under production-like
# constraints (docker run --cpus=0.1 --memory=512m), approximating
# Render's free-tier container. Repo is bind-mounted at /app; models are
# cached in a named volume at /root/.cache/huggingface.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV ENVIRONMENT=development \
    PYTHONUNBUFFERED=1
