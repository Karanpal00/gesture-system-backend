# app.py

import tempfile
from pathlib import Path
from fastapi import (
    FastAPI, UploadFile, File, Form,
    HTTPException, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from db import engine, Base, SessionLocal, Face, Gesture, GestureImage
from sqlalchemy.exc import SQLAlchemyError

import augment         # your scripts/augment.py
import gesture_infer   # inference/gesture_infer.py

# Create tables at startup
app = FastAPI(title="Gesture Control Backend")

@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Keypoints(BaseModel):
    keypoints: List[float]


def _run_training():
    # runs in background, logs go to your service logs
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "scripts" / "train_model_pt.py")],
        capture_output=False,
        text=True
    )


@app.get("/")
async def root():
    return {"status": "gesture backend is running"}


@app.post("/register_face")
async def register_face(username: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    db = SessionLocal()
    try:
        # overwrite any existing
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
        # upsert gesture
        gesture = db.query(Gesture).filter_by(name=gesture_name).first()
        if not gesture:
            gesture = Gesture(name=gesture_name, key=key)
            db.add(gesture)
            db.flush()
        else:
            gesture.key = key
            # clear old images
            db.query(GestureImage).filter_by(gesture_id=gesture.id).delete()

        # process each upload
        for upload in files:
            content = await upload.read()
            # write a temp copy for augment.py
            tmp = Path(tempfile.mkdtemp())
            orig_path = tmp / upload.filename
            orig_path.write_bytes(content)

            # augment into subdir
            aug_dir = tmp / "aug"
            aug_dir.mkdir()
            augment.augment_image(input_path=str(orig_path), output_dir=str(aug_dir))

            # save original + all augmentations in one go
            for img_file in sorted([orig_path, *aug_dir.iterdir()]):
                gi = GestureImage(
                    gesture_id=gesture.id,
                    filename=img_file.name,
                    image=img_file.read_bytes()
                )
                db.add(gi)

        db.commit()

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(500, f"Database error saving gesture: {e}")
    finally:
        db.close()

    return {
        "message": f"Gesture '{gesture_name}' registered: {len(files)} originals + augmentations."
    }


@app.post("/train_model")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training)
    return {"status": "training_started"}


@app.post("/predict")
async def predict(keypoints: Keypoints):
    kp = keypoints.keypoints
    if len(kp) != 63:
        raise HTTPException(400, "Payload must contain 63 keypoint values.")
    gesture, binding = gesture_infer.predict_from_keypoints(kp)
    return {"gesture": gesture, "keybinding": binding}


@app.get("/gestures")
async def list_gestures():
    db = SessionLocal()
    try:
        out = []
        for g in db.query(Gesture).all():
            out.append({
                "gesture": g.name,
                "key": g.key,
                "images": [img.filename for img in g.images]
            })
        return out
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn, sys, subprocess
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
