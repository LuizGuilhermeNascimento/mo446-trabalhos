"""Ligações entre pedaços fixos da figura."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import valid_affine
from .knn import knn

CORTES = (
    ((0, 0, .5, 1), (.5, 0, 1, 1)),
    ((0, 0, 1, .5), (0, .5, 1, 1)),
    ((0, 0, 1 / 3, 1), (1 / 3, 0, 2 / 3, 1), (2 / 3, 0, 1, 1)),
    ((0, 0, 1, 1 / 3), (0, 1 / 3, 1, 2 / 3), (0, 2 / 3, 1, 1)),
    ((0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1)),
)


def _dentro(p: np.ndarray, t: tuple, h: int, w: int) -> np.ndarray:
    return np.flatnonzero((p[:, 0] >= t[0] * w) & (p[:, 0] < t[2] * w) & (p[:, 1] >= t[1] * h) & (p[:, 1] < t[3] * h))


def tile_models(p0: np.ndarray, d0: np.ndarray, pm: np.ndarray, dm: np.ndarray, shape, cfg) -> list[dict]:
    """Movimentos entre pedaços fixos da figura; p0, d0 = pontos e descritores; pm, dm = os da figura espelhada."""
    h, w = shape
    if len(p0) < 6:
        return []
    min_d = max(cfg.min_dist_px, cfg.min_dist_frac * np.hypot(h, w))
    F = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    achados = []
    for corte in CORTES:
        ids = [_dentro(p0, t, h, w) for t in corte]
        idm = [_dentro(pm, t, h, w) for t in corte] if len(pm) else [np.zeros(0, int)] * len(corte)
        for a in range(len(corte)):
            for b in range(len(corte)):
                if a == b or len(ids[a]) < 3:
                    continue
                for espelho in (False, True):
                    if not espelho and b < a:
                        continue
                    pb, db = (pm[idm[b]], dm[idm[b]]) if espelho else (p0[ids[b]], d0[ids[b]])
                    if len(pb) < 3:
                        continue
                    I, D = knn(d0[ids[a]], db, 2)
                    ok = D[:, 0] < cfg.tile_ratio * D[:, 1]
                    S, T = p0[ids[a]][ok], pb[I[ok, 0]]
                    longe = np.hypot(*(S - T).T) >= min_d
                    S, T = S[longe], T[longe]
                    if len(S) < cfg.tile_min_pairs:
                        continue
                    Tf = np.c_[(w - 1) - T[:, 0], T[:, 1]].astype(np.float32) if espelho else T
                    A, inl = cv2.estimateAffinePartial2D(S, Tf, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh,
                                                         maxIters=2000, confidence=0.995, refineIters=10)
                    if A is None or inl is None:
                        continue
                    k = inl.ravel().astype(bool)
                    n = min(len(np.unique(np.round(S[k]), axis=0)), len(np.unique(np.round(T[k]), axis=0)))
                    if n < cfg.tile_min_pairs:
                        continue
                    if espelho:
                        A = (F @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
                    achados.append((n, int(len(S)), espelho, A, S[k], T[k]))
    achados.sort(key=lambda x: -x[0])
    fica = []
    for x in achados:
        chave = {(int(q[0]), int(q[1])) for q in np.round(x[4])}
        rep = False
        for y in fica:
            if y[2] != x[2]:
                continue
            comum = len(chave & y[6]) / max(len(chave), 1)
            if comum >= 0.5 and np.abs((x[4] @ y[3][:, :2].T + y[3][:, 2]) - x[5]).max() < 3 * cfg.ransac_thresh:
                rep = True
                break
        if not rep:
            fica.append(x + (chave,))
        if len(fica) >= cfg.tile_max:
            break
    models = []
    for n, tot, espelho, A, S, T, _ in fica:
        ok, geo = valid_affine(A, cfg)
        models.append(dict(A=A, S=S.astype(np.float32), T=T.astype(np.float32), inliers=2 * n, n_alive=2 * tot, inlier_ratio=n / max(tot, 1),
                           valid_geom=bool(ok), shift=float(np.median(np.hypot(*(T - S).T))),
                           spread=float(np.sqrt(np.prod(S.std(0) + 1e-6))), mirror_frac=float(espelho), tile=1.0, **geo))
    return models
