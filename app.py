# app.py

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI, Depends, UploadFile, File, Form,
    HTTPException, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import SessionLocal, engine, Base, Face, Gesture, GestureImage
from scripts.augment import augment_image
from inference.gesture_infer import predict_from_keypoints

# ─── TMP for on-disk augmentation ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TMP_DIR  = BASE_DIR / "tmp_images"
TMP_DIR.mkdir(exist_ok=True)

# ─── FastAPI setup ────────────────────────────────────────────────────────────
app = FastAPI(title="Gesture Control Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],  # your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic body model ──────────────────────────────────────────────────────
class Keypoints(BaseModel):
    keypoints: List[float]

# ─── DB dependency ────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Kick off training in background ──────────────────────────────────────────
def _run_training():
    subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "train_model_pt.py")],
        capture_output=False, text=True
    )

# ─── Startup: create tables ────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# ─── Healthcheck ──────────────────────────────────────────────────────────────
@app.get("/")
@app.head("/")
async def root():
    return {"status": "running"}

# ─── List gestures (from Postgres) ───────────────────────────────────────────
@app.get("/gestures")
async def list_gestures(db: Session = Depends(get_db)):
    out = {}
    for g in db.query(Gesture).all():
        out[g.name] = {
            "key":    g.key,
            "images": [img.filename for img in g.images]
        }
    return out

# ─── Register / replace a user’s face ─────────────────────────────────────────
@app.post("/register_face")
async def register_face(
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    data = await file.read()
    existing = db.query(Face).filter_by(username=username).first()
    if existing:
        existing.image = data
    else:
        db.add(Face(username=username, image=data))
    db.commit()
    return {"message": f"Face for '{username}' registered."}

# ─── Register / upsert a gesture + images ─────────────────────────────────────
@app.post("/register_gesture")
async def register_gesture(
    gesture_name: str    = Form(...),
    key: str             = Form(...),
    files: List[UploadFile] = File(...),
    db: Session          = Depends(get_db)
):
    # 1) Upsert the Gesture row & clear out old images if exists
    gesture = db.query(Gesture).filter_by(name=gesture_name).first()
    if gesture:
        gesture.key = key
        db.query(GestureImage).filter_by(gesture_id=gesture.id).delete()
    else:
        gesture = Gesture(name=gesture_name, key=key)
        db.add(gesture)
        db.flush()  # so gesture.id is populated

    # 2) Write originals → tmp, augment, then read back into DB
    tmp_folder = TMP_DIR / gesture_name
    shutil.rmtree(tmp_folder, ignore_errors=True)
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # save & augment
    for idx, up in enumerate(files, start=1):
        ext = Path(up.filename).suffix or ".jpg"
        name = f"orig_{idx}{ext}"
        dst  = tmp_folder / name
        dst.write_bytes(await up.read())
        augment_image(str(dst), str(tmp_folder))

    # commit each image variant into the DB
    for img_path in sorted(tmp_folder.iterdir()):
        db.add(GestureImage(
            gesture_id=gesture.id,
            filename=img_path.name,
            image=img_path.read_bytes()
        ))

    db.commit()
    return {
        "message": (
            f"Gesture '{gesture_name}' registered: "
            f"{len(files)} originals + augmentations."
        )
    }

# ─── Trigger training script (background) ─────────────────────────────────────
@app.post("/train_model")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training)
    return {"status": "training_started"}

# ─── Predict via your inference module ────────────────────────────────────────
@app.post("/predict")
async def predict(kp: Keypoints):
    if len(kp.keypoints) != 63:
        raise HTTPException(400, "Payload must contain 63 keypoint values.")
    gesture, binding = predict_from_keypoints(kp.keypoints)
    return {"gesture": gesture, "keybinding": binding}
