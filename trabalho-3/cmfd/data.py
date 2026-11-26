"""Leitura das imagens e das máscaras da competição."""
from __future__ import annotations

import cv2
import numpy as np


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"não consegui ler {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_masks(mask_path: str, shape: tuple[int, int]) -> list[np.ndarray]:
    """Lista de máscaras binárias (uma por região) de um .npy da competição."""
    if not mask_path:
        return []
    m = np.load(mask_path)
    if m.ndim == 2:
        ms = [m]
    elif m.ndim == 3:
        if m.shape[:2] == tuple(shape) and m.shape[2] != shape[0]:
            ms = [m[:, :, k] for k in range(m.shape[2])]
        else:
            ms = [m[k] for k in range(m.shape[0])]
    else:
        raise ValueError(f"máscara com formato inesperado {m.shape} em {mask_path}")
    return [(x > 0).astype(np.uint8) for x in ms]
