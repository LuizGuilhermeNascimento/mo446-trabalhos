"""Filtro de painéis: descarta o candidato cujos pontos casados caem fora dos painéis de imagem.

Os painéis vêm do SAM 3.1 congelado (frase "microscopic images", nota >= 0,3), pesos públicos, sem ajuste.
Numa foto única o SAM costuma devolver a imagem inteira como painel, e aí nada é cortado. Regra medida no
treino: fica o candidato com 50% ou mais dos pontos
de origem E de cópia dentro de painel; se o SAM não achar painel nenhum, o filtro não se aplica.
"""
from __future__ import annotations

import numpy as np

MIN_FRAC = 0.5


def load_panels(path: str) -> np.ndarray | None:
    """Arquivo de kaggle/sam_paineis.py -> máscara booleana reduzida (h, w), ou None se não há painel."""
    z = np.load(path)
    h, w = (int(v) for v in z["hw"])
    if int(z["n"]) == 0:
        return None
    return np.unpackbits(z["P"])[:h * w].reshape(h, w).astype(bool)


def _frac(P: np.ndarray, pts: np.ndarray, sx: float, sy: float) -> float:
    if len(pts) == 0:
        return 1.0
    x = np.clip((pts[:, 0] * sx).astype(int), 0, P.shape[1] - 1)
    y = np.clip((pts[:, 1] * sy).astype(int), 0, P.shape[0] - 1)
    return float(P[y, x].mean())


def off_panels(P: np.ndarray | None, S: np.ndarray, T: np.ndarray, scale: float, shape, min_frac: float = MIN_FRAC) -> bool:
    """S, T: pontos casados na escala de trabalho (scale = trabalho / original); shape = (H, W) original."""
    if P is None:
        return False
    H, W = shape
    sx, sy = P.shape[1] / W / scale, P.shape[0] / H / scale
    return min(_frac(P, np.asarray(S), sx, sy), _frac(P, np.asarray(T), sx, sy)) < min_frac
