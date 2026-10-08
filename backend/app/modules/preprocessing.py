import cv2
import numpy as np
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_DIMENSION = 1024  # resize if width or height exceeds this, keeps processing fast


def validate_file(filename: str, file_bytes: bytes) -> None:
    """Raise ValueError if the file isn't a usable image."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Use JPG or PNG.")
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB cap
        raise ValueError("File too large. Max 10MB.")


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into an RGB image array (H, W, 3), uint8.
    Handles PNGs with alpha channels by flattening onto a white background.
    """
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image. File may be corrupted.")

    # Handle alpha channel (transparent PNGs) by compositing onto white
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3] / 255.0
        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
        bgr = (bgr * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)
        img = bgr
    elif img.ndim == 2:
        # grayscale image, convert to 3-channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # OpenCV loads as BGR, convert to RGB for consistency with skimage/colorspacious downstream
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def resize_if_needed(img: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    """Resize image so neither dimension exceeds max_dim, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(filename: str, file_bytes: bytes) -> np.ndarray:
    """Full preprocessing pipeline: validate -> decode -> resize. Returns RGB uint8 array."""
    validate_file(filename, file_bytes)
    img = load_image_from_bytes(file_bytes)
    img = resize_if_needed(img)
    return img