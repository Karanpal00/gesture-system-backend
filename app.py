from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import subprocess, shutil, json, sys, os

# ─── Database setup ────────────────────────────────────────────────────────────
from db import engine, Base, SessionLocal, Face, Gesture, GestureImage
from sqlalchemy.exc import SQLAlchemyError

# Create tables on startup
app = FastAPI(title="Gesture Control Backend")

@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

# ─── CORS (explicit origins for credentials) ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5001",            # React dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data models ──────────────────────────────────────────────────────────────
class Keypoints(BaseModel):
    keypoints: List[float]

# ─── Utility to kick off training in background ───────────────────────────────
def _run_training():
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "scripts" / "train_model_pt.py")],
        capture_output=False,
        text=True
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "gesture backend is running"}

@app.post("/register_face")
async def register_face(username: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    db = SessionLocal()
    try:
        db.query(Face).filter(Face.username == username).delete()
        face = Face(username=username, image=data)
        db.add(face)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(500, "Database error saving face")
    finally:
        db.close()
    return {"message": f"Face for '{username}' registered successfully."}

@app.post("/register_gesture")
async def register_gesture(
    gesture_name: str = Form(...),
    key: str          = Form(...),
    files: List[UploadFile] = File(...)
):
    db = SessionLocal()
    try:
        gesture = db.query(Gesture).filter_by(name=gesture_name).first()
        if not gesture:
            gesture = Gesture(name=gesture_name, key=key)
            db.add(gesture)
            db.flush()
        else:
            gesture.key = key
            db.query(GestureImage).filter_by(gesture_id=gesture.id).delete()

        for upload in files:
            content = await upload.read()
            filename = Path(upload.filename).name
            gi = GestureImage(
                gesture_id=gesture.id,
                filename=filename,
                image=content
            )
            db.add(gi)
        db.commit()

        # Augmentations
        for upload in files:
            orig_bytes = await upload.read()
            tmp_dir = Path(tempfile.mkdtemp())
            in_path = tmp_dir / upload.filename
            in_path.write_bytes(orig_bytes)
            out_dir = tmp_dir / "aug"
            out_dir.mkdir()
            augment.augment_image(input_path=str(in_path), output_dir=str(out_dir))
            for aug_file in sorted(out_dir.iterdir()):
                aug_bytes = aug_file.read_bytes()
                gi = GestureImage(
                    gesture_id=gesture.id,
                    filename=aug_file.name,
                    image=aug_bytes
                )
                db.add(gi)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(500, "Database error saving gesture")
    finally:
        db.close()
    return {"message": f"Gesture '{gesture_name}' registered: {len(files)} originals + augmentations."}

@app.post("/train_model")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training)
    return {"status": "training_started"}

@app.post("/predict")
async def predict(keypoints: Keypoints):
    kp = keypoints.keypoints
    if len(kp) != 63:
        raise HTTPException(status_code=400, detail="Payload must contain 63 keypoint values.")
    gesture, binding = gesture_infer.predict_from_keypoints(kp)
    return {"gesture": gesture, "keybinding": binding}

@app.get("/gestures")
async def list_gestures():
    db = SessionLocal()
    try:
        gestures = db.query(Gesture).all()
        result = []
        for g in gestures:
            imgs = [img.filename for img in g.images]
            result.append({"gesture": g.name, "key": g.key, "images": imgs})
        return result
    finally:
        db.close()

# ─── Run locally with `uvicorn app:app --reload` ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
