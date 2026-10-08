import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb


def cluster_image_colours(image_rgb: np.ndarray, k: int = 16, random_state: int = 42):
    """
    Reduce an image to K representative colours using K-means in Lab space.

    Args:
        image_rgb: (H, W, 3) uint8 array
        k: number of clusters (representative colours)
        random_state: for reproducible results across runs

    Returns:
        cluster_centres_rgb: (K, 3) uint8 array — the K representative colours, in RGB
        pixel_labels: (H, W) int array — which cluster (0 to K-1) each pixel belongs to
    """
    h, w = image_rgb.shape[:2]

    # Convert to Lab for perceptually meaningful clustering
    img_lab = rgb2lab(image_rgb.astype(np.float64) / 255.0)
    pixels_lab = img_lab.reshape(-1, 3)  # flatten to (H*W, 3)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels_flat = kmeans.fit_predict(pixels_lab)

    centres_lab = kmeans.cluster_centers_  # (K, 3) in Lab

    # Convert cluster centres back to RGB for display/storage
    centres_rgb = lab2rgb(centres_lab.reshape(1, k, 3)).reshape(k, 3)
    centres_rgb = np.clip(centres_rgb * 255, 0, 255).astype(np.uint8)

    pixel_labels = labels_flat.reshape(h, w)

    return centres_rgb, pixel_labels


def rebuild_image_from_clusters(cluster_centres_rgb: np.ndarray, pixel_labels: np.ndarray) -> np.ndarray:
    """
    Reconstruct a full image from cluster centres and pixel labels.
    Useful to visualize/verify what the clustering looks like — every pixel
    gets replaced by its cluster's representative colour.

    Args:
        cluster_centres_rgb: (K, 3) uint8
        pixel_labels: (H, W) int array

    Returns:
        (H, W, 3) uint8 image
    """
    h, w = pixel_labels.shape
    flat_labels = pixel_labels.reshape(-1)
    reconstructed = cluster_centres_rgb[flat_labels]  # (H*W, 3)
    return reconstructed.reshape(h, w, 3)