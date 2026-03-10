# ============================================================
# PatchCore Anomaly Detection API — Dockerfile
# ============================================================
# Base image: PyTorch + CUDA (bỏ GPU thì đổi sang python:3.11-slim)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ── System deps ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ─────────────────────────────────────────────
WORKDIR /workspace

# ── Python deps (cache layer) ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app code ───────────────────────────────────────────
COPY app/ ./app/

# ── Copy model weights ─────────────────────────────────────
# Đảm bảo file .ckpt có mặt trước khi build image
COPY results/ ./results/

# ── Expose port ─────────────────────────────────────────────
EXPOSE 8000

# ── Run ─────────────────────────────────────────────────────
# --workers 1 vì PatchCore giữ VRAM — chạy multi-worker cần
# refactor sang shared memory / model pool.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
