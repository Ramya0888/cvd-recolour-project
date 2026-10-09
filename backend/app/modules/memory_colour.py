import numpy as np
import cv2


def classify_memory_colours(cluster_centres_rgb: np.ndarray) -> list:
    """
    Classify each cluster colour as a protected 'memory colour' (skin, sky, foliage)
    or 'other', using HSV range rules. Rules are intentionally loose/simple —
    good enough for a course project, and clearly documented as such.

    Args:
        cluster_centres_rgb: (K, 3) uint8 array

    Returns:
        list of K strings: "skin", "sky", "foliage", or "other" — one per cluster
    """
    k = cluster_centres_rgb.shape[0]
    # reshape to a fake 1xK image for cv2's HSV conversion
    rgb_img = cluster_centres_rgb.reshape(1, k, 3).astype(np.uint8)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)[0]  # (K, 3): H in [0,179], S,V in [0,255]

    labels = []
    for i in range(k):
        h, s, v = hsv_img[i]

        if 0 <= h <= 25 and s >= 30 and v >= 80:
            labels.append("skin")
        elif 90 <= h <= 130 and s >= 40 and v >= 80:
            labels.append("sky")
        elif 35 <= h <= 85 and s >= 40 and v >= 40:
            labels.append("foliage")
        else:
            labels.append("other")

    return labels