import cv2
from pathlib import Path
import mediapipe as mp

# ─── Setup MediaPipe Face Detection ───────────────────────────────────────────
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ─── Utilities ────────────────────────────────────────────────────────────────

def detect_faces(image_path: str):
    """
    Detect faces in an image.
    Returns a list of dicts with bounding box and score.
    Raises:
      FileNotFoundError, ValueError, RuntimeError
    """
    img_p = Path(image_path)
    if not img_p.exists() or not img_p.is_file():
        raise FileNotFoundError(f"[face_service] Image not found: {img_p}")
    
    img = cv2.imread(str(img_p))
    if img is None:
        raise ValueError(f"[face_service] Failed to load image: {img_p}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = []
    try:
        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            results = detector.process(img_rgb)
            if not results.detections:
                return []
            for det in results.detections:
                # Bounding box is relative [x,y,w,h]
                bbox = det.location_data.relative_bounding_box
                h, w, _ = img.shape
                faces.append({
                    "xmin": int(bbox.xmin * w),
                    "ymin": int(bbox.ymin * h),
                    "width": int(bbox.width * w),
                    "height": int(bbox.height * h),
                    "score": float(det.score[0])
                })
    except Exception as e:
        raise RuntimeError(f"[face_service] Face detection failed: {e}")

    return faces


def verify_single_face(image_path: str):
    """
    Ensures exactly one face in the image.
    Returns True if exactly one face detected.
    Raises:
      ValueError if zero or multiple faces.
    """
    faces = detect_faces(image_path)
    if len(faces) == 0:
        raise ValueError(f"[face_service] No faces detected in {image_path}")
    if len(faces) > 1:
        raise ValueError(f"[face_service] Multiple faces detected in {image_path}")
    return True


def compare_faces(known_image_path: str, query_image_path: str):
    """
    Placeholder for face-to-face verification.
    You can extend this to use FaceMesh landmarks or a small embedding model.
    Raises NotImplementedError by default.
    """
    raise NotImplementedError(
        "[face_service] compare_faces is not implemented — "
        "use FaceMesh or a dedicated embedding approach."
    )
