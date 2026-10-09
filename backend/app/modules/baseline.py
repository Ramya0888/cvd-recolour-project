import numpy as np
from app.modules.cvd_simulation import simulate_cvd


# Standard Daltonization error-redistribution matrix (shifts lost red-green
# info into channels a deuteranope CAN see). This is the widely-cited fixed matrix.
ERROR_MATRIX = np.array([
    [0.0, 0.0, 0.0],
    [0.7, 1.0, 0.0],
    [0.7, 0.0, 1.0],
])


def global_daltonize(image_rgb: np.ndarray, cvd_type: str = "deuteranomaly", severity: int = 100) -> np.ndarray:
    """
    Standard global Daltonization — the baseline method to compare against.
    Shifts EVERY pixel using a fixed error-redistribution matrix, with no
    clustering, pair detection, or memory-colour protection.
    """
    original = image_rgb.astype(np.float64)
    simulated = simulate_cvd(image_rgb, cvd_type, severity).astype(np.float64)

    # Error = what the CVD viewer loses
    error = original - simulated

    # Redistribute that error into visible channels
    error_shifted = error @ ERROR_MATRIX.T

    corrected = original + error_shifted
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected