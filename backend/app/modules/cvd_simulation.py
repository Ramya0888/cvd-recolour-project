import numpy as np
from colorspacious import cspace_convert


def simulate_cvd(image_rgb: np.ndarray, cvd_type: str = "deuteranomaly", severity: int = 100) -> np.ndarray:
    """
    Simulate color vision deficiency on an RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 array, values 0-255
        cvd_type: "deuteranomaly" (red-green, M cone), "protanomaly" (red-green, L cone),
                   or "tritanomaly" (blue-yellow, S cone)
        severity: 0-100, where 100 = full dichromacy (cone non-functional)

    Returns:
        (H, W, 3) uint8 array — what a CVD viewer perceives, in normal RGB for display
    """
    # colorspacious expects float values in [0, 1]
    img_float = image_rgb.astype(np.float64) / 255.0

    cvd_space = {
        "name": "sRGB1+CVD",
        "cvd_type": cvd_type,
        "severity": severity
    }

    simulated = cspace_convert(img_float, cvd_space, "sRGB1")

    # Clip because the conversion can produce slightly out-of-range values
    simulated = np.clip(simulated, 0, 1)
    simulated_uint8 = (simulated * 255).astype(np.uint8)

    return simulated_uint8


def simulate_cvd_on_colours(colours_rgb: np.ndarray, cvd_type: str = "deuteranomaly", severity: int = 100) -> np.ndarray:
    """
    Same simulation, but for a small array of individual colours (e.g. K-means cluster
    centres) rather than a full image. Used later in Module 4 (pair detection) where we
    only need to simulate ~16-32 representative colours, not every pixel.

    Args:
        colours_rgb: (N, 3) uint8 array of N colours

    Returns:
        (N, 3) uint8 array of simulated colours
    """
    # Reshape to a "fake image" of shape (1, N, 3) since cspace_convert expects image-like input
    reshaped = colours_rgb.reshape(1, -1, 3)
    simulated = simulate_cvd(reshaped, cvd_type, severity)
    return simulated.reshape(-1, 3)