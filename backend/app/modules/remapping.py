import numpy as np
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from app.modules.cvd_simulation import simulate_cvd_on_colours


def _simulated_delta_e(lab_a, lab_b, cvd_type, severity):
    """Convert two Lab colours to RGB, simulate CVD, convert back to Lab, measure ΔE."""
    rgb_pair = lab2rgb(np.array([[lab_a], [lab_b]])).reshape(2, 3)
    rgb_pair_uint8 = np.clip(rgb_pair * 255, 0, 255).astype(np.uint8)
    simulated_rgb = simulate_cvd_on_colours(rgb_pair_uint8, cvd_type, severity)
    simulated_lab = rgb2lab(simulated_rgb.astype(np.float64).reshape(1, 2, 3) / 255.0).reshape(2, 3)
    return deltaE_ciede2000(simulated_lab[0], simulated_lab[1])


def compute_cluster_shifts(
    cluster_centres_rgb: np.ndarray,
    confusable_pairs: list,
    memory_labels: list,
    cvd_type: str = "deuteranomaly",
    severity: int = 100,
    target_delta_e: float = 12.0,   # was 5.0 — clearly visible, still natural
    max_shift: float = 40.0,        # was 25.0 — headroom to reach the higher target
    protected_max_shift: float = 8.0,
):
    """
    For each confusable pair, compute a lightness shift that resolves the
    confusion, respecting memory-colour protection.

    Returns: dict {cluster_index: [delta_L, delta_a, delta_b]}
    """
    k = cluster_centres_rgb.shape[0]
    shifts = {i: np.zeros(3) for i in range(k)}

    centres_lab = rgb2lab(cluster_centres_rgb.astype(np.float64).reshape(1, k, 3) / 255.0).reshape(k, 3)

    for pair in confusable_pairs:
        i, j = pair["index_a"], pair["index_b"]
        lab_i, lab_j = centres_lab[i].copy(), centres_lab[j].copy()

        is_protected = memory_labels[i] != "other" or memory_labels[j] != "other"
        allowed_max = protected_max_shift if is_protected else max_shift

        direction = 1.0 if lab_i[0] >= lab_j[0] else -1.0
        step = 1.0
        total_shift = 0.0

        while total_shift < allowed_max:
            total_shift += step
            trial_i = lab_i.copy()
            trial_j = lab_j.copy()
            trial_i[0] = np.clip(lab_i[0] + direction * total_shift / 2, 0, 100)
            trial_j[0] = np.clip(lab_j[0] - direction * total_shift / 2, 0, 100)

            de_sim = _simulated_delta_e(trial_i, trial_j, cvd_type, severity)
            if de_sim >= target_delta_e:
                break

        # Record the shift vector (only L changes) for each cluster involved
        shifts[i] += np.array([direction * total_shift / 2, 0, 0])
        shifts[j] += np.array([-direction * total_shift / 2, 0, 0])

    return shifts


def apply_shifts_to_image(image_rgb: np.ndarray, pixel_labels: np.ndarray, shifts: dict) -> np.ndarray:
    """
    Apply per-cluster Lab shifts to the actual image, pixel by pixel —
    preserves each pixel's own texture/variation rather than flattening
    to the cluster's exact representative colour.
    """
    img_lab = rgb2lab(image_rgb.astype(np.float64) / 255.0)

    for cluster_idx, shift_vec in shifts.items():
        if np.allclose(shift_vec, 0):
            continue
        mask = (pixel_labels == cluster_idx)
        img_lab[mask] += shift_vec
        img_lab[mask, 0] = np.clip(img_lab[mask, 0], 0, 100)

    result_rgb = lab2rgb(img_lab)
    result_rgb = np.clip(result_rgb * 255, 0, 255).astype(np.uint8)
    return result_rgb