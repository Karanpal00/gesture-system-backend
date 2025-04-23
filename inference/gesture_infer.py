import onnxruntime as ort
import pickle
import numpy as np
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
ONNX_PATH  = MODEL_DIR / "gesture_clf_pt.onnx"
META_PATH  = MODEL_DIR / "meta_pt.pkl"

# ─── Load ONNX session ───────────────────────────────────────────────────
try:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"[gesture_infer] Model file not found at {ONNX_PATH}")
    sess = ort.InferenceSession(str(ONNX_PATH))
except Exception as e:
    raise RuntimeError(f"[gesture_infer] Failed to load ONNX model: {e}")

# ─── Load metadata (scaler, label_map, binding_map) ──────────────────────
try:
    if not META_PATH.exists():
        raise FileNotFoundError(f"[gesture_infer] Metadata file not found at {META_PATH}")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    scaler      = meta["scaler"]
    label_map   = meta["label_map"]
    binding_map = meta["binding_map"]
    id2label    = {v: k for k, v in label_map.items()}
except Exception as e:
    raise RuntimeError(f"[gesture_infer] Failed to load metadata: {e}")

def predict_from_keypoints(keypoints):
    """
    Perform inference on a 1D array of keypoints.
    Args:
      keypoints (list or np.ndarray): length must match scaler.n_features_in_
    Returns:
      (gesture_label: str, keybinding: str)
    Raises:
      ValueError / RuntimeError on bad input or inference failure.
    """
    # ─── Input validation ──────────────────────────────────────────────────
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"[gesture_infer] Expected 1D array, got {arr.ndim}D array")
    n_feats = getattr(scaler, "n_features_in_", None) or getattr(scaler, "mean_", None).shape[0]
    if arr.shape[0] != n_feats:
        raise ValueError(f"[gesture_infer] Expected {n_feats} features, got {arr.shape[0]}")

    # ─── Feature normalization ─────────────────────────────────────────────
    try:
        X = scaler.transform(arr.reshape(1, -1))
    except Exception as e:
        raise RuntimeError(f"[gesture_infer] Scaling failed: {e}")

    # ─── ONNX inference ───────────────────────────────────────────────────
    try:
        out = sess.run(None, {"float_input": X})
        logits = out[0]
        if logits.ndim != 2 or logits.shape[1] == 0:
            raise ValueError(f"[gesture_infer] Unexpected model output shape: {logits.shape}")
        label_idx = int(np.argmax(logits, axis=1)[0])
    except Exception as e:
        raise RuntimeError(f"[gesture_infer] ONNX inference error: {e}")

    # ─── Decode prediction ─────────────────────────────────────────────────
    try:
        gesture = id2label[label_idx]
        keybind = binding_map[gesture]
    except KeyError as e:
        raise RuntimeError(f"[gesture_infer] Post-processing lookup failed: {e}")

    return gesture, keybind
