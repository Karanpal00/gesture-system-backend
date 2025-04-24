# app.py
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI, UploadFile, File, Form,
    HTTPException, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import SessionLocal, Base, engine, Face, Gesture, GestureImage
from scripts.augment import augment_image
from inference.gesture_infer import predict_from_keypoints

# ─── Paths for on-disk augmentation (temporary) ─────────────────────────────
BASE_DIR      = Path(__file__).parent
TMP_DIR       = BASE_DIR / "tmp_images"
TMP_DIR.mkdir(exist_ok=True)

# ─── FastAPI setup ────────────────────────────────────────────────────────────
app = FastAPI(title="Gesture Control Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic model ───────────────────────────────────────────────────────────
class Keypoints(BaseModel):
    keypoints: List[float]

# ─── Background trainer ───────────────────────────────────────────────────────
def _run_training():
    subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "train_model_pt.py")],
        capture_output=False, text=True
    )

# ─── Startup: create DB tables ─────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# ─── Dependency ────────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─── Healthcheck ──────────────────────────────────────────────────────────────
@app.get("/")
@app.head("/")
async def root():
    return {"status": "running"}

# ─── List gestures (from DB) ───────────────────────────────────────────────────
@app.get("/gestures")
async def list_gestures(db: Session = next(get_db())):
    """
    Return a dict of {'gesture_name': {'key':..., 'images':[filenames...]}}
    """
    out = {}
    gestures = db.query(Gesture).all()
    for g in gestures:
        out[g.name] = {
            "key": g.key,
            "images": [img.filename for img in g.images]
        }
    return out

# ─── Register face ────────────────────────────────────────────────────────────
@app.post("/register_face")
async def register_face(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    data = await file.read()
    db = SessionLocal()
    # delete old if present
    db.query(Face).filter(Face.username == username).delete()
    db.add(Face(username=username, image=data))
    db.commit()
    db.close()
    return {"message": f"Face for '{username}' registered."}

# ─── Register gesture ─────────────────────────────────────────────────────────
@app.post("/register_gesture")
async def register_gesture(
    gesture_name: str = Form(...),
    key: str          = Form(...),
    files: List[UploadFile] = File(...)
):
    db = SessionLocal()
    # upsert Gesture
    gesture = db.query(Gesture).filter_by(name=gesture_name).first()
    if not gesture:
        gesture = Gesture(name=gesture_name, key=key)
        db.add(gesture)
        db.flush()  # populate gesture.id
    else:
        # clear old images
        db.query(GestureImage).filter_by(gesture_id=gesture.id).delete()

    saved_filenames = []
    # save originals to tmp, augment, then read back into DB
    gesture_tmp = TMP_DIR / gesture_name
    shutil.rmtree(gesture_tmp, ignore_errors=True)
    gesture_tmp.mkdir(parents=True, exist_ok=True)

    for idx, up in enumerate(files, start=1):
        ext = Path(up.filename).suffix or ".jpg"
        fname = f"orig_{idx}{ext}"
        dest = gesture_tmp / fname
        with open(dest, "wb") as f:
            f.write(await up.read())
        saved_filenames.append(fname)
        # augment into same tmp folder
        augment_image(str(dest), str(gesture_tmp))

    # load each file in tmp into the DB
    for img_path in sorted(gesture_tmp.iterdir()):
        b = img_path.read_bytes()
        db.add(GestureImage(
            gesture_id=gesture.id,
            filename=img_path.name,
            image=b
        ))

    db.commit()
    db.close()

    return {
        "message": (
            f"Gesture '{gesture_name}' registered: "
            f"{len(files)} originals + augmentations."
        )
    }

# ─── Kick off training ────────────────────────────────────────────────────────
@app.post("/train_model")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training)
    return {"status": "training_started"}

# ─── Predict ──────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(kp: Keypoints):
    if len(kp.keypoints) != 63:
        raise HTTPException(400, "Payload must contain 63 keypoint values.")
    g, b = predict_from_keypoints(kp.keypoints)
    return {"gesture": g, "keybinding": b}
