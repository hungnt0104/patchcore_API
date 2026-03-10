# PatchCore Anomaly Detection API

FastAPI backend nhận ảnh upload, chạy mô hình **PatchCore** (anomalib), và trả về:
- `anomaly_score` — điểm bất thường (image-level)
- `is_anomalous` — nhãn dự đoán
- `heatmap_b64` — heatmap dạng base64 PNG
- `overlay_b64` — ảnh gốc blend với heatmap
- `pred_mask_b64` — mặt nạ dự đoán (pixel-level)

---

## Cấu trúc dự án

```
Patchcore_API/
├── app/
│   └── main.py              ← FastAPI app
├── results/
│   └── Patchcore/MVTecAD/bottle/latest/weights/lightning/
│       └── model.ckpt       ← Model weight (Git LFS)
├── Dockerfile
├── requirements.txt
├── .gitattributes           ← Git LFS config cho *.ckpt
└── README.md
```

---

## Chạy local

```bash
# 1. Tạo virtual env
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Start server (từ thư mục gốc)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập **Swagger UI**: http://localhost:8000/docs

---

## Test nhanh với curl

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/test_image.jpg"
```

---

## Deploy bằng Docker (GPU)

```bash
# Build image
docker build -t patchcore-api .

# Chạy với GPU
docker run --gpus all -p 8000:8000 patchcore-api

# Chạy CPU only
docker run -p 8000:8000 patchcore-api
```

### Deploy lên GitHub + cloud GPU

1. **Push model weight** bằng Git LFS:
   ```bash
   git lfs install
   git add .gitattributes results/
   git commit -m "add model weights (LFS)"
   git push
   ```

2. **Build & push Docker image** lên Docker Hub / GHCR:
   ```bash
   docker build -t <username>/patchcore-api:latest .
   docker push <username>/patchcore-api:latest
   ```

3. Dùng image đó để deploy trên **RunPod / Vast.ai / GCP / AWS EC2 GPU**.

---

## API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/health` | Kiểm tra model đã load chưa |
| POST | `/predict` | Upload ảnh, nhận kết quả anomaly |
| GET | `/docs` | Swagger UI |

---

## Lưu ý

- **`IMAGE_SIZE`** trong `app/main.py` phải khớp với kích thước bạn dùng lúc train (mặc định `256×256`).
- Server dùng **1 worker** để tránh load model nhiều lần vào VRAM.
- Để scale ngang, dùng model pool hoặc serve bằng Triton Inference Server.
