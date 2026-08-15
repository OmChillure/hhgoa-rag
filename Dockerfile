FROM node:22-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY data/index ./data/index
COPY --from=ui /ui/dist ./frontend/dist

ENV PYTHONPATH=/app/src \
    VAANI_HOST=0.0.0.0 \
    VAANI_PORT=8080 \
    PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "scripts/serve.py"]
