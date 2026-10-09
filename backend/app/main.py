from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.modules.preprocessing import preprocess_image
from app.modules.cvd_simulation import simulate_cvd
import numpy as np
import cv2
import uuid
from pathlib import Path
from app.modules.clustering import cluster_image_colours, rebuild_image_from_clusters
from app.modules.pair_detection import detect_confusable_pairs
from app.modules.memory_colour import classify_memory_colours
from app.modules.remapping import compute_cluster_shifts, apply_shifts_to_image

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
@app.post("/simulate/{session_id}")
async def simulate_endpoint(session_id: str, cvd_type: str = "deuteranomaly", severity: int = 100):
    original_path = RESULTS_DIR / f"{session_id}_original.png"
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Session not found. Upload an image first.")

    img_bgr = cv2.imread(str(original_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    simulated_rgb = simulate_cvd(img_rgb, cvd_type, severity)

    save_path = RESULTS_DIR / f"{session_id}_simulated_{cvd_type}.png"
    simulated_bgr = cv2.cvtColor(simulated_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), simulated_bgr)

    return {
        "session_id": session_id,
        "cvd_type": cvd_type,
        "severity": severity,
        "message": "Simulation complete"
    }
@app.post("/cluster/{session_id}")
async def cluster_endpoint(session_id: str, k: int = 16):
    original_path = RESULTS_DIR / f"{session_id}_original.png"
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Session not found. Upload an image first.")

    img_bgr = cv2.imread(str(original_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    centres_rgb, pixel_labels = cluster_image_colours(img_rgb, k=k)

    # Save the posterized version for visual verification
    reconstructed = rebuild_image_from_clusters(centres_rgb, pixel_labels)
    save_path = RESULTS_DIR / f"{session_id}_clustered_k{k}.png"
    reconstructed_bgr = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), reconstructed_bgr)

    # Save cluster data (centres + labels) as .npy for the next module to reuse
    np.save(RESULTS_DIR / f"{session_id}_cluster_centres.npy", centres_rgb)
    np.save(RESULTS_DIR / f"{session_id}_pixel_labels.npy", pixel_labels)

    return {
        "session_id": session_id,
        "k": k,
        "cluster_centres_rgb": centres_rgb.tolist(),
        "message": "Clustering complete"
    }
@app.post("/detect-pairs/{session_id}")
async def detect_pairs_endpoint(session_id: str, cvd_type: str = "deuteranomaly", severity: int = 100):
    centres_path = RESULTS_DIR / f"{session_id}_cluster_centres.npy"
    if not centres_path.exists():
        raise HTTPException(status_code=404, detail="Run clustering first (/cluster/{session_id}).")

    cluster_centres_rgb = np.load(centres_path)

    pairs = detect_confusable_pairs(cluster_centres_rgb, cvd_type, severity)

    return {
        "session_id": session_id,
        "cvd_type": cvd_type,
        "num_clusters": int(cluster_centres_rgb.shape[0]),
        "num_confusable_pairs": len(pairs),
        "confusable_pairs": pairs
    }
@app.post("/classify-memory-colours/{session_id}")
async def memory_colour_endpoint(session_id: str):
    centres_path = RESULTS_DIR / f"{session_id}_cluster_centres.npy"
    if not centres_path.exists():
        raise HTTPException(status_code=404, detail="Run clustering first (/cluster/{session_id}).")

    cluster_centres_rgb = np.load(centres_path)
    labels = classify_memory_colours(cluster_centres_rgb)

    return {
        "session_id": session_id,
        "cluster_centres_rgb": cluster_centres_rgb.tolist(),
        "memory_colour_labels": labels
    }
@app.post("/remap/{session_id}")
async def remap_endpoint(
    session_id: str,
    cvd_type: str = "deuteranomaly",
    severity: int = 100,
    target_delta_e: float = 12.0,
):
    original_path = RESULTS_DIR / f"{session_id}_original.png"
    centres_path = RESULTS_DIR / f"{session_id}_cluster_centres.npy"
    labels_path = RESULTS_DIR / f"{session_id}_pixel_labels.npy"

    if not (original_path.exists() and centres_path.exists() and labels_path.exists()):
        raise HTTPException(status_code=404, detail="Run upload, then clustering first.")

    img_bgr = cv2.imread(str(original_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cluster_centres_rgb = np.load(centres_path)
    pixel_labels = np.load(labels_path)

    from app.modules.pair_detection import detect_confusable_pairs
    from app.modules.memory_colour import classify_memory_colours

    pairs = detect_confusable_pairs(cluster_centres_rgb, cvd_type, severity)
    memory_labels = classify_memory_colours(cluster_centres_rgb)

    shifts = compute_cluster_shifts(cluster_centres_rgb, pairs, memory_labels, cvd_type, severity, target_delta_e=target_delta_e)
    remapped_rgb = apply_shifts_to_image(img_rgb, pixel_labels, shifts)

    save_path = RESULTS_DIR / f"{session_id}_remapped.png"
    remapped_bgr = cv2.cvtColor(remapped_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), remapped_bgr)

    return {
        "session_id": session_id,
        "num_pairs_fixed": len(pairs),
        "num_protected_pairs": sum(
            1 for p in pairs if memory_labels[p["index_a"]] != "other" or memory_labels[p["index_b"]] != "other"
        ),
        "message": "Remapping complete"
    }