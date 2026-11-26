"""Nitidez do pico de correlação."""
from __future__ import annotations

import cv2
import numpy as np


def peak_sharpness(r: np.ndarray, An: np.ndarray, src: np.ndarray, W: int = 12, guard: int = 3) -> dict:
    """Correlação de r entre a origem (máscara src) e a cópia (An x) no pico e nos deslocamentos vizinhos."""
    H, Wd = r.shape
    ys, xs = np.nonzero(src)
    out = dict(pico=0.0, lado_max=0.0, lado_med=0.0, lado_std=0.0, n=int(len(ys)))
    if len(ys) < 16:
        return out
    w = cv2.warpAffine(r, np.asarray(An, np.float64), (Wd, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wp = np.pad(w, W)
    a = r[ys, xs].astype(np.float64)
    a -= a.mean()
    na = np.sqrt((a * a).sum())
    if na < 1e-6:
        return out
    C = np.zeros((2 * W + 1, 2 * W + 1))
    for dy in range(-W, W + 1):
        for dx in range(-W, W + 1):
            b = wp[ys + W + dy, xs + W + dx].astype(np.float64)
            b -= b.mean()
            nb = np.sqrt((b * b).sum())
            C[dy + W, dx + W] = (a * b).sum() / (na * nb) if nb > 1e-6 else 0.0
    d = np.maximum(np.abs(np.arange(-W, W + 1))[:, None], np.abs(np.arange(-W, W + 1))[None])
    lado = C[d >= guard]
    out.update(pico=float(C[d <= 1].max()), lado_max=float(lado.max()), lado_med=float(lado.mean()), lado_std=float(lado.std()))
    return out


def sign_peak(r: np.ndarray, An: np.ndarray, src: np.ndarray, W: int = 12, guard: int = 3) -> dict:
    """Como peak_sharpness, mas com a concordância de sinal do ruído fino, pixel a pixel."""
    H, Wd = r.shape
    ys, xs = np.nonzero(src)
    out = dict(s_pico=0.0, s_lado_max=0.0, s_lado_med=0.0, s_lado_std=0.0, s_n=0, s_frac=0.0)
    if len(ys) < 16:
        return out
    sg = np.sign(np.round(r, 3)).astype(np.float32)
    w = cv2.warpAffine(r, np.asarray(An, np.float64), (Wd, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wp = np.pad(np.sign(np.round(w, 3)).astype(np.float32), W)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    t = np.where(src[y0:y1, x0:x1], sg[y0:y1, x0:x1], 0).astype(np.float32)
    im = np.ascontiguousarray(wp[y0:y1 + 2 * W, x0:x1 + 2 * W])
    S = cv2.matchTemplate(im, t, cv2.TM_CCORR).astype(np.float64)
    N = np.rint(cv2.matchTemplate(np.abs(im), np.abs(t), cv2.TM_CCORR).astype(np.float64))
    C = np.where(N >= 16, np.rint(S) / np.maximum(N, 1), 0.0)
    d = np.maximum(np.abs(np.arange(-W, W + 1))[:, None], np.abs(np.arange(-W, W + 1))[None])
    lado = C[d >= guard]
    k = np.argmax(np.where(d <= 1, C, -2))
    out.update(s_pico=float(C.ravel()[k]), s_lado_max=float(lado.max()), s_lado_med=float(lado.mean()),
               s_lado_std=float(lado.std()), s_n=int(N.ravel()[k]), s_frac=float(N.ravel()[k] / len(ys)))
    return out
