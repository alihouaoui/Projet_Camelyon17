"""Caractéristiques « à la main » pour la référence classique : couleur et texture."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .data import ImageSource

N_BINS = 16


def patch_features(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb)
    feats = []
    for img, rng in ((rgb / 255.0, (0, 1)), (hsv, (0, 1))):
        for c in range(3):
            ch = img[..., c].ravel()
            feats.append(np.histogram(ch, bins=N_BINS, range=rng, density=True)[0])
            feats.append([ch.mean(), ch.std()])
    gray = (rgb2gray(rgb) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    feats.append(np.histogram(lbp, bins=10, range=(0, 10), density=True)[0])
    glcm = graycomatrix(gray // 8, distances=[1, 3], angles=[0, np.pi / 2], levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        feats.append(graycoprops(glcm, prop).ravel())
    return np.concatenate([np.asarray(f, dtype=np.float32).ravel() for f in feats])


def _chunk_features(src: ImageSource) -> np.ndarray:
    return np.stack([patch_features(src[i]) for i in range(len(src))])


def extract(src: ImageSource, n_jobs: int = -1, chunk: int = 2000) -> np.ndarray:
    """Calcule les caractéristiques par paquets (chaque worker relit ses images lui-même)."""
    chunks = [src.slice(s, s + chunk) for s in range(0, len(src), chunk)]
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_chunk_features)(c) for c in chunks))
