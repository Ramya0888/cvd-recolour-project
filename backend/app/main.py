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
from app.modules.metrics import compute_metrics
from fastapi.staticfiles import StaticFiles
from app.modules.baseline import global_daltonize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=str(RESULTS_DIR)), name="images")


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

@app.post("/metrics/{session_id}")
async def metrics_endpoint(session_id: str, cvd_type: str = "deuteranomaly", severity: int = 100, target_delta_e: float = 12.0):
    original_path = RESULTS_DIR / f"{session_id}_original.png"
    remapped_path = RESULTS_DIR / f"{session_id}_remapped.png"
    centres_path = RESULTS_DIR / f"{session_id}_cluster_centres.npy"

    if not (original_path.exists() and remapped_path.exists() and centres_path.exists()):
        raise HTTPException(status_code=404, detail="Run upload, cluster, and remap first.")

    original_rgb = cv2.cvtColor(cv2.imread(str(original_path)), cv2.COLOR_BGR2RGB)
    corrected_rgb = cv2.cvtColor(cv2.imread(str(remapped_path)), cv2.COLOR_BGR2RGB)
    cluster_centres_rgb = np.load(centres_path)

    from app.modules.pair_detection import detect_confusable_pairs
    from app.modules.memory_colour import classify_memory_colours
    from app.modules.remapping import compute_cluster_shifts

    pairs = detect_confusable_pairs(cluster_centres_rgb, cvd_type, severity)
    memory_labels = classify_memory_colours(cluster_centres_rgb)
    shifts = compute_cluster_shifts(cluster_centres_rgb, pairs, memory_labels, cvd_type, severity, target_delta_e=target_delta_e)

    # apply the SAME shifts to cluster centres so we can measure corrected-pair ΔE
    from skimage.color import rgb2lab, lab2rgb
    k = cluster_centres_rgb.shape[0]
    centres_lab = rgb2lab(cluster_centres_rgb.astype(np.float64).reshape(1, k, 3) / 255.0).reshape(k, 3)
    for idx, shift in shifts.items():
        centres_lab[idx] += shift
        centres_lab[idx, 0] = np.clip(centres_lab[idx, 0], 0, 100)
    corrected_centres_rgb = np.clip(lab2rgb(centres_lab.reshape(1, k, 3)).reshape(k, 3) * 255, 0, 255).astype(np.uint8)

    metrics = compute_metrics(original_rgb, corrected_rgb, pairs, corrected_centres_rgb, memory_labels, cvd_type, severity)

    return {"session_id": session_id, "metrics": metrics}

@app.post("/process/{session_id}")
async def process_endpoint(
    session_id: str,
    cvd_type: str = "deuteranomaly",
    severity: int = 100,
    k: int = 16,
    target_delta_e: float = 12.0,
    protect_memory: bool = True,
):
    original_path = RESULTS_DIR / f"{session_id}_original.png"
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Upload an image first.")

    img_rgb = cv2.cvtColor(cv2.imread(str(original_path)), cv2.COLOR_BGR2RGB)

    # 1. Cluster
    centres_rgb, pixel_labels = cluster_image_colours(img_rgb, k=k)

    # 2. Detect pairs + classify memory colours
    pairs = detect_confusable_pairs(centres_rgb, cvd_type, severity)
    memory_labels = classify_memory_colours(centres_rgb)
    if not protect_memory:
        memory_labels = ["other"] * len(memory_labels)  # toggle protection off

    # 3. Remap
    shifts = compute_cluster_shifts(centres_rgb, pairs, memory_labels, cvd_type, severity, target_delta_e=target_delta_e)
    remapped_rgb = apply_shifts_to_image(img_rgb, pixel_labels, shifts)

    # 4. Simulations (original + remapped) and baseline
    sim_original = simulate_cvd(img_rgb, cvd_type, severity)
    sim_remapped = simulate_cvd(remapped_rgb, cvd_type, severity)
    baseline_rgb = global_daltonize(img_rgb, cvd_type, severity)

    # 5. Save all outputs
    def save(name, rgb):
        cv2.imwrite(str(RESULTS_DIR / f"{session_id}_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    save("remapped", remapped_rgb)
    save("sim_original", sim_original)
    save("sim_remapped", sim_remapped)
    save("baseline", baseline_rgb)

    # 6. Metrics (reuse corrected cluster centres)
    from skimage.color import rgb2lab, lab2rgb
    centres_lab = rgb2lab(centres_rgb.astype(np.float64).reshape(1, k, 3) / 255.0).reshape(k, 3)
    for idx, shift in shifts.items():
        centres_lab[idx] += shift
        centres_lab[idx, 0] = np.clip(centres_lab[idx, 0], 0, 100)
    corrected_centres_rgb = np.clip(lab2rgb(centres_lab.reshape(1, k, 3)).reshape(k, 3) * 255, 0, 255).astype(np.uint8)
    metrics = compute_metrics(img_rgb, remapped_rgb, pairs, corrected_centres_rgb, memory_labels, cvd_type, severity)

    base = f"http://localhost:8000/images/{session_id}"
    return {
        "session_id": session_id,
        "images": {
            "original": f"{base}_original.png",
            "sim_original": f"{base}_sim_original.png",
            "remapped": f"{base}_remapped.png",
            "sim_remapped": f"{base}_sim_remapped.png",
            "baseline": f"{base}_baseline.png",
        },
        "metrics": metrics,
        "num_confusable_pairs": len(pairs),
        "confusable_pairs": pairs,
    }