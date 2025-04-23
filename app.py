from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import subprocess, shutil, json, sys, os

# ─── Setup paths ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
SCRIPTS_DIR    = BASE_DIR / "scripts"
INFERENCE_DIR  = BASE_DIR / "inference"
DATA_DIR       = BASE_DIR / "data"
GESTURES_DIR   = DATA_DIR / "gestures"
FACES_DIR      = DATA_DIR / "faces"
PROCESSED_DIR  = DATA_DIR / "processed"
GESTURES_JSON  = DATA_DIR / "gestures.json"

# Make sure Python can import our helper scripts
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(INFERENCE_DIR))

import augment                 # scripts/augment.py
import gesture_infer           # inference/gesture_infer.py

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Gesture Control Backend")

# Allow your React frontend (or any origin) to call these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Ensure storage dirs & metadata file exist ───────────────────────────────
for d in (GESTURES_DIR, FACES_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

if not GESTURES_JSON.exists():
    GESTURES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(GESTURES_JSON, "w") as f:
        json.dump({}, f)

# ─── Data models ──────────────────────────────────────────────────────────────
class Keypoints(BaseModel):
    keypoints: List[float]

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/register_face")
async def register_face(username: str = Form(...), file: UploadFile = File(...)):
    """
    Save a single face image under data/faces/<username>.<ext>
    """
    ext = Path(file.filename).suffix or ".jpg"
    out_path = FACES_DIR / f"{username}{ext}"
    with open(out_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    return {"message": f"Face for '{username}' registered successfully."}


@app.post("/register_gesture")
async def register_gesture(
    gesture_name: str = Form(...),
    key: str          = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Save uploaded images, augment them, and update gestures.json.
    """
    # 1) Save originals
    gesture_folder = GESTURES_DIR / gesture_name
    gesture_folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, upload in enumerate(files, start=1):
        ext = Path(upload.filename).suffix or ".jpg"
        filename = f"orig_{idx}{ext}"
        dest = gesture_folder / filename
        with open(dest, "wb") as buf:
            shutil.copyfileobj(upload.file, buf)
        saved.append(dest.name)

    # 2) Augment each original
    for name in saved:
        augment.augment_image(
            input_path=str(gesture_folder / name),
            output_dir=str(gesture_folder)
        )

    # 3) Update gestures.json
    with open(GESTURES_JSON, "r+") as f:
        data = json.load(f)
        data[gesture_name] = {
            "key": key,
            "images": sorted(os.listdir(gesture_folder))
        }
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    return {
        "message": (
            f"Gesture '{gesture_name}' registered: "
            f"{len(saved)} originals + augmentations."
        )
    }


@app.post("/train_model")
async def train_model():
    """
    Run your existing PyTorch training script to regenerate ONNX + meta.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "train_model_pt.py")],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {
        "message": "Model retrained successfully.",
        "logs": result.stdout.splitlines()
    }


@app.post("/predict")
async def predict(keypoints: Keypoints):
    """
    Accepts a list of 63 floats → returns {"gesture", "keybinding"}.
    """
    kp = keypoints.keypoints
    if len(kp) != 63:
        raise HTTPException(status_code=400,
            detail="Payload must contain 63 keypoint values.")
    gesture, binding = gesture_infer.predict_from_keypoints(kp)
    return {"gesture": gesture, "keybinding": binding}


@app.get("/gestures")
async def list_gestures():
    """
    Return the full gestures.json mapping.
    """
    with open(GESTURES_JSON) as f:
        return json.load(f)


# ─── Run locally with `uvicorn app:app --reload` ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
