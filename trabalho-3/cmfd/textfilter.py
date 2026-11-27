"""Filtro de texto: descarta o candidato cujos pontos casados caem em cima de letras e números."""
from __future__ import annotations

import cv2
import numpy as np


def frac(m: np.ndarray, pts: np.ndarray) -> float:
    if len(pts) == 0:
        return 0.0
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, m.shape[1] - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, m.shape[0] - 1)
    return float(m[y, x].mean())


def on_text(m: np.ndarray | None, S: np.ndarray, T: np.ndarray, scale: float, shape, thr: float = 0.5) -> bool:
    """S, T: pontos casados na escala de trabalho (scale = trabalho / original); shape = (H, W) original."""
    if m is None:
        return False
    k = np.float32([1.0, 1.0]) if m.shape == tuple(shape) else np.float32([m.shape[1] / shape[1], m.shape[0] / shape[0]])
    return max(frac(m, np.asarray(S) / scale * k), frac(m, np.asarray(T) / scale * k)) >= thr


def load_sam_text(path: str, pad: int = 4) -> np.ndarray | None:
    """Texto achado pelo SAM (kaggle/sam_paineis.py) -> máscara booleana reduzida, já com a folga, ou None."""
    z = np.load(path)
    if int(z["tn"]) == 0:
        return None
    h, w = (int(v) for v in z["thw"])
    m = np.unpackbits(z["T"])[:h * w].reshape(h, w)
    if not m.any():
        return None
    p = max(1, round(pad * h / int(z["HW"][0])))
    return cv2.dilate(m, np.ones((2 * p + 1,) * 2, np.uint8)).astype(bool)
