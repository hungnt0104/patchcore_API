"""
RunPod Serverless Handler — PatchCore Anomaly Detection
========================================================
Dùng khi deploy trên RunPod Serverless (không phải Pod thông thường).

Input JSON:
  {
    "input": {
      "image": "<base64-encoded image>",
      "mode": "overlay"   # optional: overlay | heatmap | side_by_side
    }
  }

Output JSON:
  {
    "anomaly_score": 0.85,
    "is_anomalous": true,
    "heatmap_b64": "...",
    "overlay_b64": "...",
    "pred_mask_b64": "..."
  }
"""

import io, base64, os, sys, time, logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── Compat shim (giống main.py) ──────────────────────────────────────────── #
import anomalib as _anomalib_mod
from enum import Enum as _Enum

if not hasattr(_anomalib_mod, "PrecisionType"):
    class PrecisionType(str, _Enum):
        FLOAT32  = "float32"
        FLOAT16  = "float16"
        BFLOAT16 = "bfloat16"
    _anomalib_mod.PrecisionType = PrecisionType

if not hasattr(_anomalib_mod, "TaskType"):
    try:
        from anomalib.utils.types import TaskType
        _anomalib_mod.TaskType = TaskType
    except ImportError:
        class TaskType(str, _Enum):
            CLASSIFICATION = "classification"
            DETECTION = "detection"
            SEGMENTATION = "segmentation"
        _anomalib_mod.TaskType = TaskType

# ── Config ───────────────────────────────────────────────────────────────── #
CHECKPOINT_PATH = Path("results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt")
IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load model (1 lần khi container khởi động) ───────────────────────────── #
logger.info("Loading PatchCore model …")
t0 = time.time()
from anomalib.models import Patchcore
_model = Patchcore.load_from_checkpoint(str(CHECKPOINT_PATH))
_model.eval()
_model.to(DEVICE)
logger.info("Model ready in %.1f s (device=%s)", time.time() - t0, DEVICE)


# ── Helper functions ─────────────────────────────────────────────────────── #
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0)


def to_heatmap_b64(anomaly_map: np.ndarray, size: tuple) -> str:
    a_min, a_max = anomaly_map.min(), anomaly_map.max()
    norm = (anomaly_map - a_min) / (a_max - a_min + 1e-8)
    resized = cv2.resize(norm, size)
    heat = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(heat).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def to_overlay_b64(pil_img: Image.Image, anomaly_map: np.ndarray, alpha=0.45) -> str:
    w, h = pil_img.size
    orig = np.array(pil_img.convert("RGB"))
    a_min, a_max = anomaly_map.min(), anomaly_map.max()
    norm = (anomaly_map - a_min) / (a_max - a_min + 1e-8)
    resized = cv2.resize(norm, (w, h))
    heat = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(orig, 1 - alpha, heat, alpha, 0)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── RunPod Handler ───────────────────────────────────────────────────────── #
def handler(job: dict) -> dict:
    """
    RunPod calls this function for each request.
    job["input"] contains the request payload.
    """
    job_input = job.get("input", {})

    # 1. Decode ảnh từ base64
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "Missing 'image' field (base64 string)"}

    try:
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Cannot decode image: {e}"}

    orig_w, orig_h = pil_img.size

    # 2. Preprocess + Inference
    tensor = preprocess(pil_img).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        output = _model.model(tensor)
    logger.info("Inference: %.3f s", time.time() - t0)

    # 3. Extract results
    def _np(x):
        return x.squeeze().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

    raw_score = 0.0
    if hasattr(output, "pred_score") and output.pred_score is not None:
        raw_score = float(output.pred_score.squeeze().cpu())
    elif hasattr(output, "anomaly_map") and output.anomaly_map is not None:
        raw_score = float(output.anomaly_map.max().cpu())

    # Normalize score
    anomaly_score = raw_score
    try:
        for attr in ["normalization_metrics", "_normalization_metrics"]:
            if hasattr(_model, attr):
                norm = getattr(_model, attr)
                if hasattr(norm, "min") and hasattr(norm, "max"):
                    s_min, s_max = float(norm.min.cpu()), float(norm.max.cpu())
                    if s_max > s_min:
                        anomaly_score = float(np.clip((raw_score - s_min) / (s_max - s_min), 0, 1))
                break
    except Exception:
        pass

    # Threshold
    try:
        threshold = float(_model.image_threshold.value.cpu())
    except Exception:
        threshold = 0.5
    is_anomalous = bool(anomaly_score > threshold)

    # Anomaly map
    anomaly_map_np = None
    if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
        anomaly_map_np = _np(output.anomaly_map)

    # 4. Build response
    result = {
        "anomaly_score": round(anomaly_score, 6),
        "raw_score": round(raw_score, 4),
        "is_anomalous": is_anomalous,
        "heatmap_b64": None,
        "overlay_b64": None,
        "pred_mask_b64": None,
    }

    if anomaly_map_np is not None:
        result["heatmap_b64"] = to_heatmap_b64(anomaly_map_np, (orig_w, orig_h))
        result["overlay_b64"] = to_overlay_b64(pil_img, anomaly_map_np)

    if hasattr(output, "pred_mask") and output.pred_mask is not None:
        mask = (_np(output.pred_mask) * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, (orig_w, orig_h))
        buf = io.BytesIO()
        Image.fromarray(mask_resized).save(buf, format="PNG")
        result["pred_mask_b64"] = base64.b64encode(buf.getvalue()).decode()

    return result


# ── Entry point ──────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
