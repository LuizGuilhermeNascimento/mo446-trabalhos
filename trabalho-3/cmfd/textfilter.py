"""Filtro de texto: descarta o candidato cujos pontos casados caem em cima de letras e números.

As caixas de texto vêm do Florence-2 congelado (<OCR_WITH_REGION>), pesos públicos, sem ajuste; ou, no lugar
dele, do SAM 3.1 congelado (load_sam_text), na mesma passada que acha os painéis.
Configuração escolhida no treino:
só caixas com rótulo ASCII e ao menos uma letra ou dígito, caixas maiores que 3% da imagem ignoradas,
4 px de folga, e sai o candidato com 50% ou mais dos pontos (origem ou cópia) dentro de texto.
"""
from __future__ import annotations

import cv2
import numpy as np

THR = 0.5
PAD = 4
MAX_BOX_FRAC = 0.03


def valida(lab: str) -> bool:
    """Rótulo com cara de texto de verdade: só ASCII e ao menos uma letra ou dígito (corta '٠١٠', '-', '...')."""
    t = lab.replace("</s>", "").strip()
    return t.isascii() and any(ch.isalnum() for ch in t)


def text_mask(quads, labels, H: int, W: int, pad: int = PAD) -> np.ndarray | None:
    """Máscara booleana (H, W) das caixas de texto válidas; None se não sobrar nenhuma."""
    m = np.zeros((H, W), np.uint8)
    for q, lab in zip(quads, labels):
        if not valida(lab):
            continue
        p = np.array(q, np.float32).reshape(-1, 2)
        if cv2.contourArea(p) > MAX_BOX_FRAC * H * W:  # caixa absurda (erro do modelo)
            continue
        cv2.fillPoly(m, [p.astype(np.int32)], 1)
    if not m.any():
        return None
    if pad:
        m = cv2.dilate(m, np.ones((2 * pad + 1,) * 2, np.uint8))
    return m.astype(bool)


def frac(m: np.ndarray, pts: np.ndarray) -> float:
    if len(pts) == 0:
        return 0.0
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, m.shape[1] - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, m.shape[0] - 1)
    return float(m[y, x].mean())


def on_text(m: np.ndarray | None, S: np.ndarray, T: np.ndarray, scale: float, thr: float = THR, shape=None) -> bool:
    """S, T: pontos casados na escala de trabalho (scale = trabalho / original).

    shape = (H, W) original: só é preciso quando m é uma máscara REDUZIDA (texto do SAM, load_sam_text).
    """
    if m is None:
        return False
    k = np.float32([1.0, 1.0]) if shape is None or m.shape == tuple(shape) else np.float32([m.shape[1] / shape[1], m.shape[0] / shape[0]])
    return max(frac(m, np.asarray(S) / scale * k), frac(m, np.asarray(T) / scale * k)) >= thr


def load_sam_text(path: str, pad: int = PAD) -> np.ndarray | None:
    """Texto achado pelo SAM 3.1 (kaggle/sam_paineis.py --texto) -> máscara booleana REDUZIDA já com a folga, ou None.

    É a união das instâncias pequenas (caixa <= 3% da imagem, nota >= 0,3) das frases "text", "letters", "words"
    e "numbers". A folga de `pad` px da imagem original vira max(1, round(pad x redução)) px na máscara.
    Medido no treino: com o filtro de painéis, dá a mesma nota do Florence.
    """
    z = np.load(path)
    if int(z["tn"]) == 0:
        return None
    h, w = (int(v) for v in z["thw"])
    m = np.unpackbits(z["T"])[:h * w].reshape(h, w)
    if not m.any():
        return None
    p = max(1, round(pad * h / int(z["HW"][0])))
    return cv2.dilate(m, np.ones((2 * p + 1,) * 2, np.uint8)).astype(bool)
