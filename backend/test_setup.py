import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from colorspacious import cspace_convert
from sklearn.cluster import KMeans

img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
lab = rgb2lab(img)
sim = cspace_convert(img, {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100}, "sRGB1")
print("All libraries working:", lab.shape, sim.shape)