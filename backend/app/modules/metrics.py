import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from app.modules.cvd_simulation import simulate_cvd_on_colours


def _pair_simulated_delta_e(colour_a_rgb, colour_b_rgb, cvd_type, severity):
    """Simulated-view delta E between two RGB colours."""
    pair = np.array([colour_a_rgb, colour_b_rgb], dtype=np.uint8)
    sim = simulate_cvd_on_colours(pair, cvd_type, severity)
    lab = rgb2lab(sim.astype(np.float64).reshape(1, 2, 3) / 255.0).reshape(2, 3)
    return float(deltaE_ciede2000(lab[0], lab[1]))


def compute_metrics(
    original_rgb: np.ndarray,
    corrected_rgb: np.ndarray,
    confusable_pairs: list,
    corrected_centres_rgb: np.ndarray,
    memory_labels: list,
    cvd_type: str = "deuteranomaly",
    severity: int = 100,
    visibility_threshold: float = 10.0,
):
    """Compute all evaluation metrics comparing original vs corrected image."""

    # ---- 1. DISTINGUISHABILITY (did we fix the confusion?) ----
    before_list, after_list, fixed = [], [], 0

    for pair in confusable_pairs:
        i, j = pair["index_a"], pair["index_b"]
        de_before = pair["delta_e_simulated"]
        de_after = _pair_simulated_delta_e(
            corrected_centres_rgb[i], corrected_centres_rgb[j], cvd_type, severity
        )
        before_list.append(de_before)
        after_list.append(de_after)
        if de_after >= visibility_threshold:
            fixed += 1

    num_pairs = len(confusable_pairs)
    fix_rate = (fixed / num_pairs * 100) if num_pairs else 0.0

    # ---- 2. NATURALNESS (did we keep it looking normal?) ----
    orig_f = original_rgb.astype(np.float64) / 255.0
    corr_f = corrected_rgb.astype(np.float64) / 255.0

    lab_orig = rgb2lab(orig_f)
    lab_corr = rgb2lab(corr_f)
    avg_delta_e = float(np.mean(deltaE_ciede2000(lab_orig, lab_corr)))

    ssim_score = float(ssim(rgb2gray(orig_f), rgb2gray(corr_f), data_range=1.0))

    # ---- 3. PIXELS CHANGED ----
    changed_mask = np.any(np.abs(original_rgb.astype(int) - corrected_rgb.astype(int)) > 5, axis=2)
    pct_changed = float(np.mean(changed_mask) * 100)

    # ---- 4. MEMORY-COLOUR PROTECTION CHECK ----
    protected_indices = [idx for idx, lab in enumerate(memory_labels) if lab != "other"]
    protected_count = len(protected_indices) if protected_indices else 0

    return {
        "distinguishability": {
            "num_confusable_pairs": num_pairs,
            "pairs_fixed": fixed,
            "fix_rate_percent": round(fix_rate, 1),
            "avg_delta_e_before": round(float(np.mean(before_list)), 2) if before_list else None,
            "avg_delta_e_after": round(float(np.mean(after_list)), 2) if after_list else None,
        },
        "naturalness": {
            "ssim": round(ssim_score, 4),
            "avg_delta_e_change": round(avg_delta_e, 2),
            "percent_pixels_changed": round(pct_changed, 1),
        },
        "memory_protection": {
            "protected_clusters": protected_count,
        },
    }