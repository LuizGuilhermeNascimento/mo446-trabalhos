"""Candidatos de copy-move a partir de atributos densos de um ViT pré-treinado e CONGELADO.

Nada é treinado: o DINOv2 (pesos públicos, auto-supervisionados em imagens naturais)
descreve cada patch da imagem; a auto-similaridade entre patches (excluindo a
vizinhança) com teste de razão gera pares (p, q) que seguem para o MESMO RANSAC
afim e a MESMA máscara densa por ZNCC do detector clássico. Serve para regiões
pequenas/lisas onde o SIFT não acha pontos-chave.
"""
from __future__ import annotations

import numpy as np

_MODEL = {}


def _device():
    import torch
    import os
    return "cuda" if torch.cuda.is_available() and os.environ.get("CMFD_DEEP", "") != "cpu" else "cpu"


def _load(name: str):
    if name not in _MODEL:
        import torch
        from transformers import AutoModel
        torch.set_num_threads(1)
        m = AutoModel.from_pretrained(name, use_safetensors=True)
        m.eval().to(_device())
        _MODEL[name] = m
    return _MODEL[name]


def patch_features(gray: np.ndarray, name: str = "facebook/dinov2-small", target: int = 672, patch: int | None = None):
    """Atributos L2-normalizados (gh, gw, D) e o tamanho do patch em pixels da imagem de entrada.

    Serve para DINOv2 (patch 14, só o token CLS antes dos patches) e DINOv3 (patch 16, CLS + tokens
    de registro): tamanho do patch e nº de tokens extras vêm da config.
    """
    import cv2
    import torch
    model = _load(name)
    patch = patch or int(getattr(model.config, "patch_size", 14))
    n_skip = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    h, w = gray.shape
    s = target / max(h, w)
    nh, nw = max(patch, int(round(h * s / patch)) * patch), max(patch, int(round(w * s / patch)) * patch)
    img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    x = np.repeat(img[None, :, :], 3, 0).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    x = torch.from_numpy((x - mean) / std)[None].to(_device())
    gh, gw = nh // patch, nw // patch
    with torch.no_grad():
        out = model(pixel_values=x).last_hidden_state[0]
    if out.shape[0] != gh * gw + n_skip:
        raise ValueError(f"tokens inesperados: {out.shape[0]} != {gh}*{gw} + {n_skip}")
    out = out[n_skip:]  # CLS (+ registros) vêm antes dos patches
    f = out.reshape(gh, gw, -1).float().cpu().numpy().astype(np.float32)
    f /= np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6
    return f, (h / nh * patch, w / nw * patch)


def deep_matches(gray: np.ndarray, name: str = "facebook/dinov2-small", target: int = 672, ratio: float = 0.9,
                 min_sim: float = 0.6, excl: int = 2, flat_std: float = 3.0):
    """Pares (P, Q) em coordenadas da imagem: vizinho mais parecido fora da vizinhança, com teste de razão."""
    f, (py, px) = patch_features(gray, name, target)
    gh, gw, D = f.shape
    F = f.reshape(-1, D)
    sim = F @ F.T
    yy, xx = np.divmod(np.arange(gh * gw), gw)
    near = (np.abs(yy[:, None] - yy[None, :]) <= excl) & (np.abs(xx[:, None] - xx[None, :]) <= excl)
    sim[near] = -1
    # patches sem textura (fundo liso) não são consultados: casam com qualquer coisa
    import cv2

    big = cv2.resize(gray.astype(np.float32), (gw * 8, gh * 8), interpolation=cv2.INTER_AREA)
    loc_std = big.reshape(gh, 8, gw, 8).std(axis=(1, 3)).ravel()

    top2 = np.argpartition(-sim, 2, axis=1)[:, :2]
    s2 = np.take_along_axis(sim, top2, 1)
    order = np.argsort(-s2, 1)
    top2 = np.take_along_axis(top2, order, 1)
    s2 = np.take_along_axis(s2, order, 1)
    d1, d2 = 1 - s2[:, 0], 1 - s2[:, 1]
    ok = (s2[:, 0] >= min_sim) & (d1 < ratio * np.maximum(d2, 1e-6)) & (loc_std > flat_std)
    i = np.flatnonzero(ok)
    j = top2[i, 0]
    keep = i < j  # cada par uma vez
    i, j = i[keep], j[keep]
    cy, cx = (yy + 0.5) * py, (xx + 0.5) * px
    P = np.stack([cx[i], cy[i]], 1).astype(np.float32)
    Q = np.stack([cx[j], cy[j]], 1).astype(np.float32)
    return P, Q, float(max(px, py))
