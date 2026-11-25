"""Recortes de um candidato para a rede pequena que compara origem e cópia (só testado no treino; fora do fluxo final).

Para cada candidato (região de origem + movimento origem -> cópia), 6 canais de 64 x 64:
  0, 1  contexto: a caixa da origem com folga, reduzida ou ampliada para 64 x 64, e o trecho da cópia alinhado a ela
  2, 3  detalhe: 64 x 64 pixels da imagem no tamanho original, centrados na origem, e o trecho alinhado da cópia
  4, 5  ruído fino (imagem menos ela mesma borrada) dos dois recortes de detalhe
As medidas do verificador resumem a região em números; aqui a rede vê os pixels dos dois lados.
"""
from __future__ import annotations

import cv2
import numpy as np

LADO = 64


def recortes(full: np.ndarray, rfull: np.ndarray, An: np.ndarray, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(uint8 (4, 64, 64): contexto origem, contexto cópia, detalhe origem, detalhe cópia; float16 (2, 64, 64): ruído)."""
    ys, xs = np.nonzero(src)
    if not len(ys):
        return np.zeros((4, LADO, LADO), np.uint8), np.zeros((2, LADO, LADO), np.float16)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    lado = max(24.0, 1.5 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    A3 = np.vstack([np.asarray(An, np.float64), [0.0, 0.0, 1.0]])

    def par(img, s, interp):
        T = np.array([[s, 0.0, cx - s * LADO / 2.0], [0.0, s, cy - s * LADO / 2.0], [0.0, 0.0, 1.0]])  # recorte -> imagem
        a = cv2.warpAffine(img, T[:2], (LADO, LADO), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(img, (A3 @ T)[:2], (LADO, LADO), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return a, b
    s = lado / LADO
    c0, c1 = par(full, s, cv2.INTER_AREA if s > 1 else cv2.INTER_LINEAR)
    d0, d1 = par(full, 1.0, cv2.INTER_LINEAR)
    r0, r1 = par(rfull, 1.0, cv2.INTER_LINEAR)
    return np.stack([c0, c1, d0, d1]).astype(np.uint8), np.stack([r0, r1]).astype(np.float16)
