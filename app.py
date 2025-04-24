# app.py
import os, sys, subprocess, shutil, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from db import engine, Base
from scripts.augment import augment_image
from inference.gesture_infer import predict_from_keypoints

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
GESTURES_DIR  = DATA_DIR / "gestures"
FACES_DIR     = DATA_DIR / "faces"
PROCESSED_DIR = DATA_DIR / "processed"
GESTURES_JSON = DATA_DIR / "gestures.json"

# ─── FastAPI setup ────────────────────────────────────────────────────────────
app = FastAPI(title="Gesture Control Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],  # your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models & Utilities ───────────────────────────────────────────────────────
class Keypoints(BaseModel):
    keypoints: List[float]

def _run_training():
    subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "train_model_pt.py")],
        capture_output=False, text=True
    )

# ─── Startup: create dirs, JSON & DB tables ───────────────────────────────────
@app.on_event("startup")
def on_startup():
    # filesystem
    for d in (GESTURES_DIR, FACES_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not GESTURES_JSON.exists():
        GESTURES_JSON.parent.mkdir(parents=True, exist_ok=True)
        GESTURES_JSON.write_text("{}")
    # database
    Base.metadata.create_all(bind=engine)

# ─── Routes (GET + HEAD combined) ─────────────────────────────────────────────
@app.get("/")
@app.head("/")
async def root():
    return {"status": "gesture backend is running"}

@app.get("/gestures")
@app.head("/gestures")
async def list_gestures():
    return json.loads(GESTURES_JSON.read_text())

# ─── Face & Gesture registration ─────────────────────────────────────────────
@app.post("/register_face")
async def register_face(username: str = Form(...), file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    out = FACES_DIR / f"{username}{ext}"
    with out.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)
    return {"message": f"Face for '{username}' registered successfully."}

@app.post("/register_gesture")
async def register_gesture(
    gesture_name: str = Form(...),
    key: str          = Form(...),
    files: List[UploadFile] = File(...)
):
    folder = GESTURES_DIR / gesture_name
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, up in enumerate(files, 1):
        ext = Path(up.filename).suffix or ".jpg"
        dest = folder / f"orig_{i}{ext}"
        with dest.open("wb") as buf:
            shutil.copyfileobj(up.file, buf)
        saved.append(dest.name)
        augment_image(str(dest), str(folder))

    data = json.loads(GESTURES_JSON.read_text())
    data[gesture_name] = {"key": key, "images": sorted(saved + 
                                        [f for f in folder.iterdir()
                                         if f.name not in saved])}
    GESTURES_JSON.write_text(json.dumps(data, indent=2))
    return {
      "message": f"Gesture '{gesture_name}' registered: {len(saved)} originals + augmentations."
    }

# ─── Train & Predict ─────────────────────────────────────────────────────────
@app.post("/train_model")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training)
    return {"status": "training_started"}

@app.post("/predict")
async def predict(kp: Keypoints):
    if len(kp.keypoints) != 63:
        raise HTTPException(400, "Payload must contain 63 keypoint values.")
    g, b = predict_from_keypoints(kp.keypoints)
    return {"gesture": g, "keybinding": b}
