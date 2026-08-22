# Single-stage image. The app is pure Python with no build step and no static asset
# pipeline, so a multi-stage build would add complexity without saving anything.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Dependencies first so the layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/pools.json ./data/pools.json

# Videos are NOT copied into the image. They are served from object storage in
# deployment, and baking roughly a gigabyte of media into the image would make every
# deploy slow and blow past the image size limits of the free hosting tiers.

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000
# Two workers: the app is IO bound and the free tier gives one shared CPU.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
