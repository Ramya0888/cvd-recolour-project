from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.modules.preprocessing import preprocess_image
import numpy as np
import cv2
import uuid
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        img_rgb = preprocess_image(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save preprocessed image so later steps/modules can reuse it
    session_id = str(uuid.uuid4())
    save_path = RESULTS_DIR / f"{session_id}_original.png"
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "width": img_rgb.shape[1],
        "height": img_rgb.shape[0],
        "message": "Image preprocessed successfully"
    }
