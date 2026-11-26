"""Ligações por LightGlue no lugar do casamento por vizinho mais próximo (só testado no treino; fora do fluxo final)."""
from __future__ import annotations

import cv2
import numpy as np

_M = {}
GRADES = (2, 4)
MAX_PONTOS = 4096


MAX_ALIKED = 2048


def _modelos(kind: str, cfg=None):
    """Extrator + LightGlue."""
    key = (kind, getattr(cfg, "contrast_threshold", None), getattr(cfg, "edge_threshold", None))
    if key not in _M:
        import torch
        from lightglue import ALIKED, SIFT, LightGlue
        torch.set_num_threads(1)
        if kind == "sift":
            ext = SIFT(max_num_keypoints=MAX_PONTOS, detection_threshold=cfg.contrast_threshold, edge_threshold=cfg.edge_threshold).eval()
        else:
            ext = ALIKED(max_num_keypoints=MAX_ALIKED, detection_threshold=0.0).eval()
        if ("lg", kind) not in _M:
            _M[("lg", kind)] = LightGlue(features=kind).eval()
        _M[key] = (ext, _M[("lg", kind)])
    return _M[key]


def _extrai(gray: np.ndarray, kind: str, cfg=None) -> dict:
    import torch
    ext, _ = _modelos(kind, cfg)
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)[None]
    if kind != "sift":
        t = t.repeat(3, 1, 1)
    with torch.no_grad():
        return ext.extract(t, resize=None)


def _sub(f: dict, idx):
    import torch
    idx = torch.as_tensor(idx, dtype=torch.long)
    out = {}
    for k, v in f.items():
        out[k] = v[:, idx] if (hasattr(v, "ndim") and v.ndim >= 2 and v.shape[1] == f["keypoints"].shape[1] and k != "image_size") else v
    return out


def _casa(kind: str, f0: dict, f1: dict):
    import torch
    mt = _M[("lg", kind)]
    if f0["keypoints"].shape[1] < 2 or f1["keypoints"].shape[1] < 2:
        return np.zeros((0, 2), int)
    with torch.no_grad():
        r = mt({"image0": f0, "image1": f1})
    return r["matches"][0].cpu().numpy()


def self_matches_lg(gray: np.ndarray, cfg, kind: str) -> dict:
    """Mesma saída de cmfd.detector.self_matches: P, Q (pontos ligados) e mirror."""
    h, w = gray.shape
    f = _extrai(gray, kind, cfg)
    kp = f["keypoints"][0].cpu().numpy()
    n0 = len(kp)
    out = dict(n_kp=int(n0), n_kp_mirror=0, P=np.zeros((0, 2), np.float32), Q=np.zeros((0, 2), np.float32), mirror=np.zeros(0, bool))
    if n0 < 4:
        return out
    min_d = max(cfg.min_dist_px, cfg.min_dist_frac * np.hypot(h, w))
    P, Q, M, seen = [], [], [], set()
    for g in GRADES:
        cel = np.minimum((kp[:, 1] * g / h).astype(int), g - 1) * g + np.minimum((kp[:, 0] * g / w).astype(int), g - 1)
        ids = [np.nonzero(cel == c)[0] for c in range(g * g)]
        for a in range(g * g):
            for b in range(a + 1, g * g):
                if len(ids[a]) < 2 or len(ids[b]) < 2:
                    continue
                for i, j in _casa(kind, _sub(f, ids[a]), _sub(f, ids[b])):
                    i, j = int(ids[a][i]), int(ids[b][j])
                    if (i, j) in seen or np.hypot(*(kp[i] - kp[j])) < min_d:
                        continue
                    seen.add((i, j))
                    P.append(kp[i]); Q.append(kp[j]); M.append(False)
    if cfg.mirror:
        fm = _extrai(cv2.flip(gray, 1), kind, cfg)
        km = fm["keypoints"][0].cpu().numpy().copy()
        out["n_kp_mirror"] = int(len(km))
        if len(km) >= 2:
            mm = _casa(kind, f, fm)
            km[:, 0] = (w - 1) - km[:, 0]
            for i, j in mm:
                if np.hypot(*(kp[i] - km[j])) < min_d:
                    continue
                P.append(kp[i]); Q.append(km[j]); M.append(True)
    if P:
        out.update(P=np.float32(P), Q=np.float32(Q), mirror=np.array(M))
    return out
