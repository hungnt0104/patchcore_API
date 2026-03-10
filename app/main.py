"""
PatchCore Anomaly Detection FastAPI Backend
==========================================
Nhận ảnh upload → chạy PatchCore → trả về:
  - anomaly_score (float)
  - is_anomalous (bool)
  - heatmap (base64 PNG)
  - overlay (base64 PNG — ảnh gốc + heatmap)
"""

import io
import base64
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")   # phải đặt trước import pyplot để tránh lỗi GUI thread
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# --------------------------------------------------------------------------- #
# Compatibility shim — PHẢI đặt trước mọi import anomalib
# --------------------------------------------------------------------------- #
# Model được train bằng anomalib 1.x có PrecisionType / TaskType ở top-level
# anomalib namespace. Anomalib 2.x đã xóa/chuyển các class này.
# torch.load dùng pickle để deserialize .ckpt → phải inject đúng class
# vào đúng namespace TRƯỚC khi load, và class phải ở MODULE LEVEL
# (không được định nghĩa trong function) để pickle's find_class hoạt động.
import anomalib as _anomalib_compat
from enum import Enum as _Enum

if not hasattr(_anomalib_compat, "PrecisionType"):
    class PrecisionType(str, _Enum):  # noqa: N801
        """Backward-compat stub for anomalib 1.x PrecisionType."""
        FLOAT32  = "float32"
        FLOAT16  = "float16"
        BFLOAT16 = "bfloat16"
    _anomalib_compat.PrecisionType = PrecisionType  # type: ignore[attr-defined]

if not hasattr(_anomalib_compat, "TaskType"):
    try:
        from anomalib.utils.types import TaskType          # v1 path
        _anomalib_compat.TaskType = TaskType               # type: ignore[attr-defined]
    except ImportError:
        class TaskType(str, _Enum):                        # type: ignore[no-redef]
            """Backward-compat stub for anomalib 1.x TaskType."""
            CLASSIFICATION = "classification"
            DETECTION      = "detection"
            SEGMENTATION   = "segmentation"
        _anomalib_compat.TaskType = TaskType               # type: ignore[attr-defined]

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CHECKPOINT_PATH = Path("results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt")
IMAGE_SIZE = (256, 256)   # kích thước đầu vào của model (thay nếu train khác)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Lifespan — load model 1 lần khi khởi động
# --------------------------------------------------------------------------- #
model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load PatchCore model vào bộ nhớ khi server start."""
    logger.info("🚀 Loading PatchCore model from %s …", CHECKPOINT_PATH)
    t0 = time.time()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {CHECKPOINT_PATH}")

    # ── Import model và load checkpoint ─────────────────────────────────────── #
    # (Shim đã được inject ở module level phía trên — trước import anomalib)
    from anomalib.models import Patchcore

    model = Patchcore.load_from_checkpoint(str(CHECKPOINT_PATH))
    model.eval()
    model.to(DEVICE)

    # Lưu reference đến ImageBatch class (anomalib v2.x)
    try:
        from anomalib.data import ImageBatch
        model_state["ImageBatch"] = ImageBatch
    except ImportError:
        model_state["ImageBatch"] = None  # anomalib v1.x fallback

    model_state["model"] = model
    logger.info("✅ Model loaded in %.2f s  (device=%s)", time.time() - t0, DEVICE)

    yield  # ← server chạy trong đây

    logger.info("🛑 Shutting down — releasing model …")
    model_state.clear()


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="PatchCore Anomaly Detection API",
    description="Upload một ảnh để phát hiện bất thường (anomaly) bằng PatchCore.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helper: tiền xử lý ảnh → tensor
# --------------------------------------------------------------------------- #
def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Resize, normalize (ImageNet mean/std), trả về tensor [1, 3, H, W].
    Giống hệt transform mà anomalib dùng mặc định.
    """
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_img.convert("RGB"))  # [3, H, W]
    return tensor.unsqueeze(0)                  # [1, 3, H, W]


# --------------------------------------------------------------------------- #
# Helper: tensor anomaly_map → heatmap PNG (base64)
# --------------------------------------------------------------------------- #
def anomaly_map_to_heatmap_b64(anomaly_map: np.ndarray,
                               original_size: tuple[int, int]) -> str:
    """Trả về chuỗi base64 của ảnh heatmap (colormap jet)."""
    # Normalize về [0, 1]
    a_min, a_max = anomaly_map.min(), anomaly_map.max()
    if a_max > a_min:
        norm = (anomaly_map - a_min) / (a_max - a_min)
    else:
        norm = np.zeros_like(anomaly_map)

    # Resize về kích thước ảnh gốc
    norm_resized = cv2.resize(norm, (original_size[0], original_size[1]))

    # Áp dụng colormap jet
    heatmap_bgr = cv2.applyColorMap(
        (norm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    buf = io.BytesIO()
    Image.fromarray(heatmap_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Helper: overlay heatmap lên ảnh gốc
# --------------------------------------------------------------------------- #
def overlay_heatmap_b64(pil_img: Image.Image,
                        anomaly_map: np.ndarray,
                        alpha: float = 0.5) -> str:
    """Blend ảnh gốc + heatmap, trả về base64 PNG."""
    orig_w, orig_h = pil_img.size
    orig_np = np.array(pil_img.resize((orig_w, orig_h)).convert("RGB"))

    a_min, a_max = anomaly_map.min(), anomaly_map.max()
    if a_max > a_min:
        norm = (anomaly_map - a_min) / (a_max - a_min)
    else:
        norm = np.zeros_like(anomaly_map)

    norm_resized = cv2.resize(norm, (orig_w, orig_h))
    heatmap_bgr = cv2.applyColorMap(
        (norm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(orig_np, 1 - alpha, heatmap_rgb, alpha, 0)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Endpoint: health check
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["System"])
async def health_check():
    """Kiểm tra server và model đã sẵn sàng chưa."""
    return {
        "status": "ok",
        "model_loaded": "model" in model_state,
        "device": DEVICE,
    }


# --------------------------------------------------------------------------- #
# Endpoint: predict
# --------------------------------------------------------------------------- #
@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Ảnh cần kiểm tra (JPG/PNG)")):
    """
    ### Phát hiện bất thường trên ảnh upload

    **Trả về JSON:**
    ```json
    {
        "anomaly_score": 0.85,
        "is_anomalous": true,
        "pred_mask_b64": "<base64 PNG>",
        "heatmap_b64":   "<base64 PNG>",
        "overlay_b64":   "<base64 PNG>"
    }
    ```
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model chưa load xong, thử lại sau.")

    # ── 1. Đọc ảnh ────────────────────────────────────────────────────────── #
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {e}")

    orig_w, orig_h = pil_img.size
    logger.info("Received image %s  size=(%d, %d)", file.filename, orig_w, orig_h)

    # ── 2. Tiền xử lý ─────────────────────────────────────────────────────── #
    tensor = preprocess_image(pil_img).to(DEVICE)   # [1, 3, H, W]

    # ── 3. Inference ──────────────────────────────────────────────────────── #
    # Gọi thẳng model.model (PatchcoreModel — inner torch module)
    # với tensor [1, 3, H, W]. KHÔNG squeeze vì BatchNorm cần 4D input.
    model = model_state["model"]
    t0 = time.time()
    with torch.no_grad():
        output = model.model(tensor)   # tensor shape: [1, C, H, W]
    logger.info("Inference done in %.3f s", time.time() - t0)


    # ── 4. Trích xuất kết quả ─────────────────────────────────────────────── #
    # output có thể là ImageBatch hoặc dict, ta xử lý cả 2 case
    def _to_float(x):
        if isinstance(x, torch.Tensor):
            return float(x.squeeze().cpu().item())
        return float(x)

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.squeeze().cpu().numpy()
        return np.array(x)

    # Anomaly score (image-level raw score từ PatchcoreModel)
    if hasattr(output, "pred_score") and output.pred_score is not None:
        raw_score = _to_float(output.pred_score)
    elif hasattr(output, "anomaly_map") and output.anomaly_map is not None:
        raw_score = float(output.anomaly_map.max().cpu())
    else:
        raw_score = 0.0

    # ── Normalize raw score về [0, 1] ────────────────────────────────────── #
    # PatchcoreModel trả raw distance score. Anomalib engine normalize bằng
    # min/max của validation set. Ta lấy lại stats từ normalization_metrics.
    anomaly_score = raw_score
    try:
        norm = None
        for attr in ["normalization_metrics", "_normalization_metrics",
                     "image_normalization_metric"]:
            if hasattr(model, attr):
                norm = getattr(model, attr)
                break
        if norm is not None and hasattr(norm, "min") and hasattr(norm, "max"):
            s_min = float(norm.min.cpu())
            s_max = float(norm.max.cpu())
            if s_max > s_min:
                anomaly_score = float(np.clip(
                    (raw_score - s_min) / (s_max - s_min), 0.0, 1.0
                ))
                logger.info("Normalized score: %.4f (raw=%.4f)", anomaly_score, raw_score)
        else:
            logger.warning("No normalization stats found, returning raw score %.4f", raw_score)
    except Exception as e:
        logger.warning("Normalization failed (%s), using raw score %.4f", e, raw_score)

    # is_anomalous: threshold từ lúc train
    if hasattr(output, "pred_label") and output.pred_label is not None:
        is_anomalous = bool(_to_float(output.pred_label))
    elif isinstance(output, dict) and "pred_label" in output:
        is_anomalous = bool(_to_float(output["pred_label"]))
    else:
        try:
            threshold = float(model.image_threshold.value.cpu())
        except Exception:
            threshold = 0.5
        is_anomalous = anomaly_score > threshold
        logger.info("Threshold: %.4f | score: %.4f | anomalous: %s",
                    threshold, anomaly_score, is_anomalous)




    # Anomaly map (pixel-level heatmap)
    anomaly_map_np: np.ndarray | None = None
    if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
        anomaly_map_np = _to_numpy(output.anomaly_map)
    elif isinstance(output, dict) and "anomaly_map" in output:
        anomaly_map_np = _to_numpy(output["anomaly_map"])

    # Predicted mask
    pred_mask_np: np.ndarray | None = None
    if hasattr(output, "pred_mask") and output.pred_mask is not None:
        pred_mask_np = _to_numpy(output.pred_mask)
    elif isinstance(output, dict) and "pred_mask" in output:
        pred_mask_np = _to_numpy(output["pred_mask"])

    # ── 5. Tạo ảnh kết quả (base64) ───────────────────────────────────────── #
    heatmap_b64 = None
    overlay_b64 = None
    pred_mask_b64 = None

    if anomaly_map_np is not None:
        heatmap_b64 = anomaly_map_to_heatmap_b64(anomaly_map_np, (orig_w, orig_h))
        overlay_b64 = overlay_heatmap_b64(pil_img, anomaly_map_np, alpha=0.45)

    if pred_mask_np is not None:
        mask_resized = cv2.resize(
            (pred_mask_np * 255).astype(np.uint8), (orig_w, orig_h)
        )
        buf = io.BytesIO()
        Image.fromarray(mask_resized).save(buf, format="PNG")
        pred_mask_b64 = base64.b64encode(buf.getvalue()).decode()

    # ── 6. Trả về JSON ────────────────────────────────────────────────────── #
    return JSONResponse(content={
        "anomaly_score": round(anomaly_score, 6),
        "is_anomalous": is_anomalous,
        "heatmap_b64": heatmap_b64,
        "overlay_b64": overlay_b64,
        "pred_mask_b64": pred_mask_b64,
    })


# --------------------------------------------------------------------------- #
# Endpoint: /predict/visualize  — trả về overlay image trực tiếp
# --------------------------------------------------------------------------- #
@app.post("/predict/visualize", tags=["Inference"],
          response_class=StreamingResponse)
async def predict_visualize(
    file: UploadFile = File(..., description="Ảnh cần kiểm tra (JPG/PNG)"),
    mode: str = "overlay",   # "overlay" | "heatmap" | "side_by_side"
):
    """
    Trả về **ảnh PNG** (không phải JSON) để xem trực tiếp trong browser:
    - `mode=overlay`     — ảnh gốc blend với heatmap
    - `mode=heatmap`     — heatmap thuần (colormap jet)
    - `mode=side_by_side`— ảnh gốc | heatmap | overlay xếp ngang
    """
    # Tái sử dụng logic /predict
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")

    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {e}")

    tensor = preprocess_image(pil_img).to(DEVICE)
    with torch.no_grad():
        output = model_state["model"].model(tensor)

    orig_w, orig_h = pil_img.size

    def _to_np(x):
        return x.squeeze().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

    anomaly_map_np = None
    if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
        anomaly_map_np = _to_np(output.anomaly_map)

    if anomaly_map_np is None:
        raise HTTPException(status_code=500, detail="Model không trả về anomaly_map.")

    if mode == "heatmap":
        img_b64 = anomaly_map_to_heatmap_b64(anomaly_map_np, (orig_w, orig_h))
        result_bytes = base64.b64decode(img_b64)
    elif mode == "side_by_side":
        orig_np = np.array(pil_img)
        # heatmap
        a_min, a_max = anomaly_map_np.min(), anomaly_map_np.max()
        norm = (anomaly_map_np - a_min) / (a_max - a_min + 1e-8)
        norm_r = cv2.resize(norm, (orig_w, orig_h))
        heat_bgr = cv2.applyColorMap((norm_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        # overlay
        blend = cv2.addWeighted(orig_np, 0.55, heat_rgb, 0.45, 0)
        canvas = np.concatenate([orig_np, heat_rgb, blend], axis=1)
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        result_bytes = buf.getvalue()
    else:  # overlay (default)
        img_b64 = overlay_heatmap_b64(pil_img, anomaly_map_np, alpha=0.45)
        result_bytes = base64.b64decode(img_b64)

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="result_{mode}.png"'},
    )


# --------------------------------------------------------------------------- #
# Demo page: /demo
# --------------------------------------------------------------------------- #
_DEMO_HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>PatchCore Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }
  body { background: #0f0f1a; color: #e0e0e0; min-height: 100vh; padding: 2rem; }
  h1 { text-align: center; font-size: 2rem; color: #7dd3fc;
       text-shadow: 0 0 20px #7dd3fc55; margin-bottom: 0.25rem; }
  p.sub { text-align: center; color: #94a3b8; margin-bottom: 2rem; }
  .card { background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 12px;
          padding: 1.5rem; max-width: 960px; margin: 0 auto 1.5rem; }
  .upload-area { border: 2px dashed #4c4c8a; border-radius: 8px; padding: 2rem;
                 text-align: center; cursor: pointer; transition: .2s;
                 background: #13132a; }
  .upload-area:hover { border-color: #7dd3fc; background: #1a2040; }
  input[type=file] { display: none; }
  label.btn, button.btn {
    display: inline-block; margin-top: 1rem; padding: .65rem 1.8rem;
    background: linear-gradient(135deg, #3b82f6, #7c3aed);
    color: #fff; border: none; border-radius: 8px; cursor: pointer;
    font-size: 1rem; font-weight: 600; transition: .2s; }
  label.btn:hover, button.btn:hover { opacity: .85; transform: translateY(-1px); }
  select { background: #1a1a2e; color: #e0e0e0; border: 1px solid #4c4c8a;
           border-radius: 6px; padding: .4rem .8rem; font-size: .95rem; margin-left: .5rem; }
  .results { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
  .result-box { background: #13132a; border-radius: 8px; overflow: hidden; text-align: center; }
  .result-box img { width: 100%; display: block; }
  .result-box .label { padding: .5rem; font-size: .8rem; color: #94a3b8; }
  .score-bar { background: #13132a; border-radius: 8px; padding: 1rem 1.5rem; margin-top: 1rem; }
  .score-row { display: flex; justify-content: space-between; margin-bottom: .5rem; }
  .badge { padding: .3rem .8rem; border-radius: 20px; font-weight: 700; font-size: .9rem; }
  .badge.ok { background: #14532d; color: #4ade80; }
  .badge.ng { background: #7f1d1d; color: #f87171; }
  .bar-bg { background: #2d2d4e; border-radius: 4px; height: 8px; flex: 1; margin: 0 1rem;
             align-self: center; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .5s;
               background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); }
  #preview { max-width: 260px; border-radius: 8px; margin: 1rem auto 0; display: none; }
  .spin { display: none; text-align: center; margin: 1rem; font-size: 1.5rem; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .sp { display: inline-block; animation: spin 1s linear infinite; }
</style>
</head>
<body>
<h1>🔍 PatchCore Anomaly Detection</h1>
<p class="sub">Upload ảnh để phát hiện bất thường (anomaly)</p>

<div class="card">
  <div class="upload-area" onclick="document.getElementById('fileInput').click()">
    <div style="font-size:2.5rem">📂</div>
    <p style="margin:.5rem 0">Kéo thả hoặc <strong>click</strong> để chọn ảnh</p>
    <p style="font-size:.8rem;color:#64748b">JPG / PNG</p>
  </div>
  <input type="file" id="fileInput" accept="image/*" onchange="onFileChange(event)">
  <img id="preview">
  <div style="margin-top:1rem;text-align:center">
    Mode:
    <select id="modeSelect">
      <option value="side_by_side">Side by Side</option>
      <option value="overlay">Overlay</option>
      <option value="heatmap">Heatmap only</option>
    </select>
    <button class="btn" onclick="runPredict()" style="margin-left:.5rem">▶ Phân tích</button>
  </div>
  <div class="spin" id="spinner"><span class="sp">⏳</span> Đang chạy mô hình…</div>
</div>

<div class="card" id="resultCard" style="display:none">
  <div class="score-bar">
    <div class="score-row">
      <span style="font-weight:600">Anomaly Score</span>
      <span id="scoreTxt" style="font-size:1.2rem;font-weight:700"></span>
      <span id="badge" class="badge"></span>
    </div>
    <div style="display:flex;align-items:center;margin-top:.4rem">
      <span style="font-size:.75rem;color:#64748b">0</span>
      <div class="bar-bg"><div class="bar-fill" id="barFill"></div></div>
      <span style="font-size:.75rem;color:#64748b">1</span>
    </div>
  </div>
  <img id="resultImg" style="width:100%;border-radius:8px;margin-top:1rem">
</div>

<script>
let selectedFile = null;
function onFileChange(e) {
  selectedFile = e.target.files[0];
  if (!selectedFile) return;
  const url = URL.createObjectURL(selectedFile);
  const prev = document.getElementById('preview');
  prev.src = url; prev.style.display = 'block';
  document.getElementById('resultCard').style.display = 'none';
}
async function runPredict() {
  if (!selectedFile) { alert('Vui lòng chọn ảnh trước!'); return; }
  const mode = document.getElementById('modeSelect').value;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('resultCard').style.display = 'none';
  // - Gọi /predict để lấy score
  const fd1 = new FormData(); fd1.append('file', selectedFile);
  const fd2 = new FormData(); fd2.append('file', selectedFile);
  const [jsonRes, imgRes] = await Promise.all([
    fetch('/predict', { method: 'POST', body: fd1 }),
    fetch(`/predict/visualize?mode=${mode}`, { method: 'POST', body: fd2 })
  ]);
  const data = await jsonRes.json();
  const imgBlob = await imgRes.blob();
  document.getElementById('spinner').style.display = 'none';
  // Hiện score
  const score = data.anomaly_score;
  const display = score > 1 ? score.toFixed(2) : (score * 100).toFixed(1) + '%';
  document.getElementById('scoreTxt').textContent = display;
  const pct = Math.min(score > 1 ? Math.min(score / 30, 1) : score, 1) * 100;
  document.getElementById('barFill').style.width = pct + '%';
  const badge = document.getElementById('badge');
  if (data.is_anomalous) {
    badge.textContent = '❌ NG — Bất thường'; badge.className = 'badge ng';
  } else {
    badge.textContent = '✅ OK — Bình thường'; badge.className = 'badge ok';
  }
  // Hiện ảnh
  document.getElementById('resultImg').src = URL.createObjectURL(imgBlob);
  document.getElementById('resultCard').style.display = 'block';
}
</script>
</body>
</html>
"""

@app.get("/demo", response_class=HTMLResponse, tags=["Demo"])
async def demo_page():
    """Trang demo trực quan: upload ảnh và xem heatmap/overlay ngay trong browser."""
    return HTMLResponse(content=_DEMO_HTML)


if __name__ == "__main__":
    import sys, os, uvicorn
    # Đảm bảo thư mục gốc project luôn được Python tìm thấy
    # dù chạy từ app/ hay từ Patchcore_API/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # Tắt reload để tránh lỗi subprocess trên Windows
        workers=1,
    )
