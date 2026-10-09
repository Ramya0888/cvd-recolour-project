import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from app.modules.cvd_simulation import simulate_cvd_on_colours


def detect_confusable_pairs(
    cluster_centres_rgb: np.ndarray,
    cvd_type: str = "deuteranomaly",
    severity: int = 100,
    normal_threshold: float = 10.0,
    simulated_threshold: float = 5.0,
):
    """
    Find pairs of cluster colours that look clearly different normally,
    but become confusable (too similar) under CVD simulation.

    Args:
        cluster_centres_rgb: (K, 3) uint8 array of representative colours
        cvd_type, severity: passed to the CVD simulator
        normal_threshold: min Delta E in normal vision to count as "clearly different"
        simulated_threshold: max Delta E in simulated vision to count as "confusable"

    Returns:
        list of dicts, each describing one confusable pair:
        {
            "index_a": int, "index_b": int,
            "colour_a": [r,g,b], "colour_b": [r,g,b],
            "delta_e_normal": float,
            "delta_e_simulated": float
        }
    """
    k = cluster_centres_rgb.shape[0]

    # Convert original colours to Lab (for normal-vision Delta E)
    centres_norm = cluster_centres_rgb.astype(np.float64) / 255.0
    lab_normal = rgb2lab(centres_norm.reshape(1, k, 3)).reshape(k, 3)

    # Simulate CVD on the cluster colours, then convert simulated colours to Lab too
    simulated_rgb = simulate_cvd_on_colours(cluster_centres_rgb, cvd_type, severity)
    simulated_norm = simulated_rgb.astype(np.float64) / 255.0
    lab_simulated = rgb2lab(simulated_norm.reshape(1, k, 3)).reshape(k, 3)

    confusable_pairs = []

    for i in range(k):
        for j in range(i + 1, k):
            de_normal = deltaE_ciede2000(lab_normal[i], lab_normal[j])
            de_simulated = deltaE_ciede2000(lab_simulated[i], lab_simulated[j])

            if de_normal >= normal_threshold and de_simulated <= simulated_threshold:
                confusable_pairs.append({
                    "index_a": int(i),
                    "index_b": int(j),
                    "colour_a": cluster_centres_rgb[i].tolist(),
                    "colour_b": cluster_centres_rgb[j].tolist(),
                    "delta_e_normal": float(de_normal),
                    "delta_e_simulated": float(de_simulated),
                })

    # Sort by severity: biggest drop in distinguishability first
    confusable_pairs.sort(key=lambda p: p["delta_e_normal"] - p["delta_e_simulated"], reverse=True)

    return confusable_pairs