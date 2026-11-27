"""Ligações sem teste de razão, confirmadas uma a uma pela correlação do ruído fino."""
from __future__ import annotations

import cv2
import numpy as np

from .knn import _torch_device


def links_from_knn(kn: dict, cfg, k: int, shape) -> dict:
    """Ligações de cada ponto aos k vizinhos mais parecidos (sem teste de razão), dos vizinhos já calculados."""
    h, w = shape
    p0, pts, I, n0 = kn["p0"], kn["pts"], kn["I"], kn["n0"]
    if I.shape[1] < k + 1:
        raise ValueError(f"vizinhos guardados ({I.shape[1]}) insuficientes para k = {k}")
    i = np.repeat(np.arange(n0), I.shape[1])
    j = I.ravel()
    keep = i != j
    keep &= np.cumsum(keep.reshape(n0, -1), 1).ravel() <= k
    i, j = i[keep], j[keep]
    mir = j >= n0
    P, Qo = p0[i], pts[j]
    longe = np.hypot(*(P - Qo).T) >= max(cfg.min_dist_px, cfg.min_dist_frac * np.hypot(h, w))
    d = Qo - P
    sentido = mir | (d[:, 0] > 0) | ((d[:, 0] == 0) & (d[:, 1] > 0))
    P2, Qo2 = np.where(sentido[:, None], P, Qo), np.where(sentido[:, None], Qo, P)
    a, b = np.where(sentido, i, j), np.where(sentido, j, i)
    _, un = np.unique(np.stack([a, b], 1), axis=0, return_index=True)
    sel = np.zeros(len(i), bool)
    sel[un] = True
    sel &= longe
    return dict(P=P2[sel], Qo=Qo2[sel], mir=mir[sel])


def confirma_ligacoes(gray_full: np.ndarray, P: np.ndarray, Q: np.ndarray, M: np.ndarray, radius: int = 8,
                       sigma: float = 1.0, search: float = 1.0, lote: int = 20000) -> np.ndarray:
    """Correlação do ruído fino entre o recorte em P e o recorte em Q (busca de meio em meio pixel), por ligação."""
    import torch
    import torch.nn.functional as F
    if len(P) == 0:
        return np.zeros(0, np.float32)
    tb = _torch_device()
    dev = tb[1] if tb is not None else torch.device("cpu")
    G = gray_full.astype(np.float32)
    H, W = G.shape
    r = torch.from_numpy(G - cv2.GaussianBlur(G, (0, 0), sigma)).to(dev)[None, None]
    n1 = 2 * radius + 1
    d = torch.arange(-radius, radius + 1, dtype=torch.float32, device=dev)
    g = torch.arange(-(radius + search) * 2, (radius + search) * 2 + 1, dtype=torch.float32, device=dev) / 2
    ng = len(g)
    passos = [x / 2 for x in range(int(-search * 2), int(search * 2) + 1)]

    def amostra(cx, cy, ox, oy):
        x = cx[:, None, None] + ox[:, None, :]
        y = cy[:, None, None] + oy[None, :, None]
        x, y = torch.broadcast_tensors(x, y)
        grid = torch.stack([(2 * x + 1) / W - 1, (2 * y + 1) / H - 1], -1).reshape(1, -1, x.shape[2], 2)
        return F.grid_sample(r, grid, mode="bilinear", padding_mode="reflection", align_corners=False).reshape(x.shape).double()

    out = []
    with torch.no_grad():
        for b0 in range(0, len(P), lote):
            p = torch.from_numpy(np.ascontiguousarray(P[b0:b0 + lote], np.float32)).to(dev)
            q = torch.from_numpy(np.ascontiguousarray(Q[b0:b0 + lote], np.float32)).to(dev)
            sg = torch.from_numpy(np.where(M[b0:b0 + lote], -1.0, 1.0).astype(np.float32)).to(dev)
            n = len(p)
            a = amostra(p[:, 0], p[:, 1], d[None].expand(n, -1), d)
            a = a - a.mean((1, 2), keepdim=True)
            saa = (a * a).sum((1, 2))
            big = amostra(q[:, 0], q[:, 1], sg[:, None] * g[None], g)
            best = torch.full((n,), -1.0, dtype=torch.float64, device=dev)
            for sinal in (1.0, -1.0):
                sub = torch.nonzero(sg == sinal).ravel()
                if not len(sub):
                    continue
                bg, aa, sa = big[sub], a[sub], saa[sub]
                melhor = torch.full((len(sub),), -1.0, dtype=torch.float64, device=dev)
                for oy in passos:
                    y0 = int(round((search + oy) * 2))
                    for ox in passos:
                        x0 = int(round((search + sinal * ox) * 2))
                        b = bg[:, y0:y0 + 2 * n1:2, x0:x0 + 2 * n1:2]
                        sb = b.sum((1, 2))
                        sbb = (b * b).sum((1, 2)) - sb * sb / (n1 * n1)
                        sab = (aa * b).sum((1, 2))
                        melhor = torch.maximum(melhor, sab / torch.clamp(torch.sqrt(sa * torch.clamp(sbb, min=0.0)), min=1e-6))
                best[sub] = melhor
            best[torch.sqrt(saa) < 1e-3 * (n1 * n1) ** 0.5] = -1.0
            out.append(best.float().cpu().numpy())
    return np.concatenate(out)


def verified_models(gray_full: np.ndarray, shape, s: float, cfg, kn: dict, min_pairs: int = 2) -> list[dict]:
    """Ligações confirmadas pelo ruído, agrupadas por deslocamento; cada grupo vira um candidato de translação."""
    m = links_from_knn(kn, cfg, cfg.vlinks_k, shape)
    if len(m["P"]) < min_pairs:
        return []
    P, Q, M = m["P"], m["Qo"], m["mir"]
    ncc = confirma_ligacoes(gray_full, P / s, Q / s, M, cfg.vlinks_radius)
    ok = ncc >= cfg.vlinks_thr
    P, Q, M, ncc = P[ok], Q[ok], M[ok], ncc[ok]
    if len(P) < min_pairs:
        return []
    t = Q - np.where(M[:, None], np.c_[-P[:, :1], P[:, 1:]], P)
    tol = max(3.0 * cfg.ransac_thresh, 0.01 * np.hypot(*shape))
    livre = np.ones(len(P), bool)
    rw = cfg.vlinks_radius * s
    corners = np.array([[-rw, -rw], [rw, -rw], [rw, rw], [-rw, rw]], np.float32)
    out = []
    while len(out) < cfg.vlinks_max and livre.sum() >= min_pairs:
        idx = np.flatnonzero(livre)
        viz = np.zeros(len(idx), np.int64)
        for b in range(0, len(idx), 2000):
            x = t[idx[b:b + 2000]]
            viz[b:b + 2000] = ((np.hypot(x[:, None, 0] - t[None, idx, 0], x[:, None, 1] - t[None, idx, 1]) < tol)
                               & (M[idx[b:b + 2000], None] == M[None, idx])).sum(1)
        c = idx[int(viz.argmax())]
        same = livre & (M == M[c]) & (np.hypot(*(t - t[c]).T) < tol)
        livre &= ~same
        n = int(same.sum())
        if n < min_pairs:
            break
        mm = bool(M[c])
        tv = np.median(t[same], 0)
        A = np.array([[-1.0 if mm else 1.0, 0.0, tv[0]], [0.0, 1.0, tv[1]]])
        S = (P[same][:, None, :] + corners[None]).reshape(-1, 2).astype(np.float32)
        T = (A[:, :2] @ S.T + A[:, 2:]).T.astype(np.float32)
        out.append(dict(A=A, S=S, T=T, inliers=n, n_alive=int(len(P)), inlier_ratio=n / max(len(P), 1), valid_geom=True,
                        shift=float(np.hypot(*(T - S).T).mean()), spread=float(np.sqrt(np.prod(S.std(0) + 1e-6))),
                        mirror_frac=float(mm), scale=1.0, aniso=1.0, reflect=mm, vseed=1.0, vseed_ncc=float(ncc[same].mean())))
    return out
