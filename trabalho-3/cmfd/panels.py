"""Separação simples de painéis por fundo claro (usada na ablação da ideia 1).

Mesma receita do 6º lugar do Kaggle: o que não é quase branco vira primeiro
plano; um fechamento morfológico une o conteúdo de cada painel; cada contorno
externo grande é um painel. Uma figura de painel único devolve um só painel.
"""
from __future__ import annotations

import cv2
import numpy as np


def panel_map(gray: np.ndarray, white: int = 240, close_frac: float = 0.01,
              min_area_frac: float = 0.01) -> np.ndarray:
    """Mapa (H, W) int16 com o índice do painel de cada pixel (-1 = fundo)."""
    h, w = gray.shape
    fg = (gray < white).astype(np.uint8)
    k = max(3, int(round(close_frac * max(h, w))) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = np.full((h, w), -1, np.int16)
    idx = 0
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        if bw * bh < min_area_frac * h * w:
            continue
        out[y:y + bh, x:x + bw] = np.where(out[y:y + bh, x:x + bw] < 0, idx, out[y:y + bh, x:x + bw])
        idx += 1
    if idx == 0:  # nada encontrado: a imagem inteira é um painel
        out[:] = 0
    return out


def caixas_classicas(gray: np.ndarray, white: int = 240, close_frac: float = 0.004, min_area_frac: float = 0.0005) -> np.ndarray:
    """Caixas (x0, y0, x1, y1 em frações da figura) dos blocos não brancos: fotos, faixas de blot, gráficos.
    Mesma receita de panel_map, com fechamento menor e área mínima menor, para faixas finas não se juntarem nem
    sumirem. Serve de alternativa quando o SAM não devolve fotos (figuras de blot). Só testado no treino."""
    h, w = gray.shape
    fg = (gray < white).astype(np.uint8)
    k = max(3, int(round(close_frac * max(h, w))) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = [(x / w, y / h, (x + bw) / w, (y + bh) / h) for x, y, bw, bh, _ in stats[1:] if bw * bh >= min_area_frac * h * w]
    return np.asarray(out, np.float32).reshape(-1, 4)
