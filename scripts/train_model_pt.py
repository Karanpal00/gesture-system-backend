#!/usr/bin/env python
"""
Train a lightweight MLP on hand-keypoint vectors with PyTorch,
export to ONNX, and save scaler + label_map + binding_map.
If no CSVs exist under data/processed/, it will first pull
all images from the PostgreSQL database, extract MediaPipe
hand landmarks, and produce a single CSV under data/processed/.
Run: python scripts/train_model_pt.py
"""

import os, pathlib
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"  # writable cache for matplotlib
import sys
import logging
import pickle
import json

# Database imports
from db import SessionLocal, Gesture, GestureImage

# Ensure onnx is available
try:
    import onnx
except ModuleNotFoundError:
    print("[train_model] ERROR: 'onnx' package is not installed.")
    print("   Install it via:  pip install onnx")
    sys.exit(1)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cv2
import mediapipe as mp

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="[train_model] %(levelname)s: %(message)s"
)

# Paths
BASE_DIR      = pathlib.Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR     = BASE_DIR / "models"

# MediaPipe Hands
mp_hands     = mp.solutions.hands
HANDS_CONFIG = dict(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Model definition
class KeypointMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64,  n_classes)
        )
    def forward(self, x): return self.net(x)


def generate_dataset_from_db() -> pathlib.Path:
    logging.info("No CSVs found—generating dataset from DB images.")
    session = SessionLocal()
    records = []
    hands_detector = mp_hands.Hands(**HANDS_CONFIG)

    for gesture in session.query(Gesture).all():
        binding = gesture.key
        for img in gesture.images:
            arr = np.frombuffer(img.image, dtype=np.uint8)
            img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_cv is None:
                logging.warning(f"Failed to decode {img.filename}; skipping")
                continue
            results = hands_detector.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                logging.warning(f"No hand detected in {img.filename}; skipping")
                continue
            lm = results.multi_hand_landmarks[0]
            kp = [coord for pt in lm.landmark for coord in (pt.x, pt.y, pt.z)]
            rec = {"label": gesture.name, "binding": binding}
            rec.update({f"kp{i}": float(k) for i, k in enumerate(kp)})
            records.append(rec)

    hands_detector.close()
    session.close()

    if not records:
        raise RuntimeError("No valid keypoint records generated from any DB image.")

    df = pd.DataFrame.from_records(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DIR / "gestures_dataset.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Generated dataset CSV from DB: {out_csv} ({len(df)} rows)")
    return out_csv


def load_data():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csvs = list(PROCESSED_DIR.glob("*.csv"))
    if not csvs:
        csvs = [generate_dataset_from_db()]

    frames = []
    for csv in csvs:
        logging.info(f"Reading CSV {csv}")
        frames.append(pd.read_csv(csv))
    df = pd.concat(frames, ignore_index=True)

    if "label" not in df.columns or "binding" not in df.columns:
        raise ValueError("CSV must contain 'label' and 'binding' columns.")

    features = [c for c in df.columns if c.startswith("kp")]
    X = df[features].to_numpy(dtype=np.float32)
    y_lbl = df["label"].to_numpy()
    classes = sorted(df["label"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.vectorize(label_map.get)(y_lbl)

    binding_map = {row["label"]: row["binding"] for _, row in df.iterrows()}
    return X, y, label_map, binding_map


def main():
    try:
        logging.info("Loading data …")
        X, y, label_map, binding_map = load_data()
        n_samples, n_features = X.shape
        n_classes = len(label_map)
        logging.info(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        Xtr_t = torch.from_numpy(Xtr_s)
        ytr_t = torch.from_numpy(ytr).long()
        Xte_t = torch.from_numpy(Xte_s)
        yte_t = torch.from_numpy(yte).long()

        model     = KeypointMLP(in_dim=n_features, n_classes=n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs, batch_size = 60, 64
        logging.info("Starting training …")
        for ep in range(1, epochs+1):
            perm = torch.randperm(Xtr_t.size(0))
            for i in range(0, Xtr_t.size(0), batch_size):
                idx = perm[i : i+batch_size]
                optimizer.zero_grad()
                loss = criterion(model(Xtr_t[idx]), ytr_t[idx])
                loss.backward(); optimizer.step()

            if ep % 10 == 0 or ep == epochs:
                with torch.no_grad():
                    preds = model(Xte_t).argmax(dim=1)
                    acc   = (preds == yte_t).float().mean().item() * 100
                logging.info(f"Epoch {ep}/{epochs} — val acc: {acc:.2f}%")

        pt_path   = MODEL_DIR / "gesture_clf_pt.pt"
        onnx_path = MODEL_DIR / "gesture_clf_pt.onnx"
        meta_path = MODEL_DIR / "meta_pt.pkl"

        logging.info(f"Saving PyTorch model → {pt_path}")
        torch.save(model.state_dict(), pt_path)

        logging.info(f"Exporting ONNX model → {onnx_path}")
        try:
            dummy = torch.randn(1, n_features)
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=["float_input"],
                output_names=["output"],
                dynamic_axes={"float_input": {0: "batch"}}
            )
        except Exception as e:
            logging.error(f"ONNX export failed: {e}")
            sys.exit(1)

        logging.info(f"Saving metadata → {meta_path}")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "scaler":      scaler,
                "label_map":   label_map,
                "binding_map": binding_map
            }, f)

        logging.info("✅ Training & export complete.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
