"""Candidatos por voto de Hough (Lowe 2004), sem teste de razão (só testado no treino; fora do fluxo final).

Motivo: nas cópias pequenas o vizinho mais parecido de um ponto É a cópia, mas o teste de
razão do g2NN descarta a ligação, e com 2 ligações o movimento afim (6 graus de liberdade) sai deformado.
Aqui cada ponto liga aos seus k vizinhos mais parecidos, sem razão. Cada ligação já propõe um movimento completo
(deslocamento, giro e tamanho, pela orientação e escala do próprio ponto SIFT) e vota numa caixa desse espaço.
Caixas com votos suficientes viram candidatos: ajusta-se uma semelhança (4 graus de liberdade) às ligações da caixa.
A confirmação fica com as etapas de sempre (máscara por ZNCC, correlação do ruído, seleção).
"""
from __future__ import annotations

import cv2
import numpy as np

from .knn import knn as _knn


def _sift_full(gray: np.ndarray, cfg):
    sift = cv2.SIFT_create(nfeatures=cfg.max_keypoints, contrastThreshold=cfg.contrast_threshold,
                           edgeThreshold=cfg.edge_threshold)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        z = np.zeros(0, np.float32)
        return np.zeros((0, 2), np.float32), z, z, np.zeros((0, 128), np.float32)
    pts = np.float32([k.pt for k in kps])
    des = des.astype(np.float32)
    if cfg.rootsift:
        des /= des.sum(1, keepdims=True) + 1e-7
        des = np.sqrt(des)
    return pts, np.float32([k.size for k in kps]), np.float32([k.angle for k in kps]), des


def hough_matches(gray: np.ndarray, cfg, k: int = 2) -> dict:
    """Ligações sem razão. P = ponto na imagem; Q = vizinho (na imagem ou na imagem espelhada, coordenadas DELA);
    Qo = Q nas coordenadas da imagem; mir = vizinho veio da imagem espelhada; ds, da = razão de escala e giro (graus)."""
    h, w = gray.shape
    p0, s0, a0, d0 = _sift_full(gray, cfg)
    vazio = dict(P=np.zeros((0, 2), np.float32), Q=np.zeros((0, 2), np.float32), Qo=np.zeros((0, 2), np.float32),
                 mir=np.zeros(0, bool), ds=np.zeros(0, np.float32), da=np.zeros(0, np.float32), n_kp=len(p0))
    n0 = len(p0)
    if n0 < 4:
        return vazio
    pts, siz, ang, des = p0, s0, a0, d0
    if cfg.mirror:
        pm, sm, am, dm = _sift_full(cv2.flip(gray, 1), cfg)
        if len(pm):
            pts, siz, ang, des = np.vstack([p0, pm]), np.r_[s0, sm], np.r_[a0, am], np.vstack([d0, dm])
    I, _ = _knn(d0, des, min(k + 1, len(des)))
    i = np.repeat(np.arange(n0), I.shape[1])
    j = I.ravel()
    keep = i != j
    # só os k primeiros vizinhos que não são o próprio ponto
    ordem = np.cumsum(keep.reshape(n0, -1), 1).ravel()
    keep &= ordem <= k
    i, j = i[keep], j[keep]
    mir = j >= n0
    Q = pts[j]
    Qo = Q.copy()
    Qo[mir, 0] = (w - 1) - Qo[mir, 0]
    P = p0[i]
    min_d = max(cfg.min_dist_px, cfg.min_dist_frac * np.hypot(h, w))
    longe = np.hypot(*(P - Qo).T) >= min_d
    # sem espelho, a ligação (i, j) e a (j, i) são a mesma: fica um sentido só (deslocamento para a direita)
    d = Qo - P
    sentido = mir | (d[:, 0] > 0) | ((d[:, 0] == 0) & (d[:, 1] > 0))
    P2, Q2, Qo2 = np.where(sentido[:, None], P, Qo), np.where(sentido[:, None], Q, P), np.where(sentido[:, None], Qo, P)
    si, sj = np.where(sentido, siz[i], siz[j]), np.where(sentido, siz[j], siz[i])
    ai, aj = np.where(sentido, ang[i], ang[j]), np.where(sentido, ang[j], ang[i])
    a, b = np.where(sentido, i, j), np.where(sentido, j, i)
    _, un = np.unique(np.stack([a, b], 1), axis=0, return_index=True)
    sel = np.zeros(len(i), bool)
    sel[un] = True
    sel &= longe
    return dict(P=P2[sel], Q=Q2[sel], Qo=Qo2[sel], mir=mir[sel], ds=(sj / np.maximum(si, 1e-6))[sel],
                da=((aj - ai) % 360.0)[sel], n_kp=n0)


def hough_models(gray: np.ndarray, cfg, k: int = 2, min_pairs: int = 3, bin_frac: float = 0.01, max_models: int = 12) -> list[dict]:
    """Movimentos candidatos (mesmo formato de detector.fit_models) a partir das caixas de Hough mais votadas."""
    from .detector import _valid_affine
    h, w = gray.shape
    m = hough_matches(gray, cfg, k)
    n = len(m["P"])
    if n < min_pairs:
        return []
    bw = max(8.0, bin_frac * np.hypot(h, w))
    d = m["Q"] - m["P"]  # deslocamento do próprio ponto (para o espelho, nas coordenadas da imagem espelhada)
    coord = np.stack([d[:, 0] / bw, d[:, 1] / bw, m["da"] / 30.0, np.log2(np.maximum(m["ds"], 1e-6))], 1)
    base = np.floor(coord - 0.5).astype(np.int64)  # cada ligação vota nas 2 caixas mais próximas de cada dimensão
    votos, quem = [], []
    for o in range(16):
        off = np.array([(o >> b) & 1 for b in range(4)])
        c = base + off
        c[:, 2] %= 12
        code = ((c[:, 0] + 5000) * 10001 + (c[:, 1] + 5000)) * 12 * 64 * 2 + (c[:, 2] * 64 + (c[:, 3] + 32)) * 2 + m["mir"]
        votos.append(code)
        quem.append(np.arange(n))
    votos, quem = np.concatenate(votos), np.concatenate(quem)
    o = np.argsort(votos, kind="stable")
    votos, quem = votos[o], quem[o]
    ini = np.flatnonzero(np.r_[True, votos[1:] != votos[:-1]])
    tam = np.diff(np.r_[ini, len(votos)])
    usado = np.zeros(n, bool)
    F = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    models = []
    for b in np.argsort(-tam, kind="stable"):
        if tam[b] < min_pairs or len(models) >= max_models:
            break
        idx = quem[ini[b]:ini[b] + tam[b]]
        idx = idx[~usado[idx]]
        if len(idx) < min_pairs:
            continue
        P, Q = m["P"][idx], m["Q"][idx]
        A, inl = cv2.estimateAffinePartial2D(P, Q, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh,
                                             maxIters=2000, confidence=0.995, refineIters=10)
        if A is None or inl is None:
            continue
        inl = inl.ravel().astype(bool)
        if inl.sum() < min_pairs:
            continue
        # o ajuste vale para todas as ligações ainda livres, não só as da caixa
        livres = np.flatnonzero(~usado & (m["mir"] == m["mir"][idx[0]]))
        err = np.hypot(*((m["P"][livres] @ A[:, :2].T + A[:, 2]) - m["Q"][livres]).T)
        dentro = livres[err <= cfg.ransac_thresh]
        usado[dentro] = True
        if m["mir"][idx[0]]:
            A = (F @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
        S, T = m["P"][dentro], m["Qo"][dentro]
        ok, geo = _valid_affine(A, cfg)
        models.append(dict(A=A, S=S, T=T, inliers=2 * len(dentro), n_alive=2 * n, inlier_ratio=len(dentro) / n,
                           valid_geom=bool(ok), shift=float(np.median(np.hypot(*(T - S).T))),
                           spread=float(np.sqrt(np.prod(S.std(0) + 1e-6))), mirror_frac=float(m["mir"][idx[0]]),
                           hough=1.0, **geo))
    return models


def links_from_knn(kn: dict, cfg, k: int, shape) -> dict:
    """As mesmas ligações de hough_matches (P, Qo, mir), a partir dos pontos e vizinhos já calculados por
    detector.self_matches (kn = dict(p0, pts, I, n0)), sem refazer o SIFT nem a busca de vizinhos."""
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
    P, Qo = p0[i], pts[j]  # pts do espelho já estão nas coordenadas da imagem
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
    """A mesma medida de detector.verify_pairs (correlação do ruído fino entre o recorte em P e o recorte em Q, com
    busca de meio em meio pixel até ±search; espelho horizontal quando M), calculada com o torch: na GPU quando há
    (como a busca de vizinhos, cmfd/knn.py), senão na CPU, com a mesma conta. Uma amostragem bilinear só em torno de
    Q; os 25 deslocamentos são fatias dela. Difere de verify_pairs na 3ª casa (lá o cv2.remap arredonda as
    coordenadas em 1/32 px)."""
    import torch
    import torch.nn.functional as F

    from .knn import _torch_device
    if len(P) == 0:
        return np.zeros(0, np.float32)
    tb = _torch_device()
    dev = tb[1] if tb is not None else torch.device("cpu")
    G = gray_full.astype(np.float32)
    H, W = G.shape
    r = torch.from_numpy(G - cv2.GaussianBlur(G, (0, 0), sigma)).to(dev)[None, None]
    n1 = 2 * radius + 1
    d = torch.arange(-radius, radius + 1, dtype=torch.float32, device=dev)
    g = torch.arange(-(radius + search) * 2, (radius + search) * 2 + 1, dtype=torch.float32, device=dev) / 2  # meio em meio pixel
    ng = len(g)
    passos = [x / 2 for x in range(int(-search * 2), int(search * 2) + 1)]

    def amostra(cx, cy, ox, oy):  # (n, len(oy), len(ox)): r em (cx + ox, cy + oy), bilinear, borda refletida
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
            big = amostra(q[:, 0], q[:, 1], sg[:, None] * g[None], g)  # no espelho a grade corre ao contrário em x
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
                        sab = (aa * b).sum((1, 2))  # a tem média zero
                        melhor = torch.maximum(melhor, sab / torch.clamp(torch.sqrt(sa * torch.clamp(sbb, min=0.0)), min=1e-6))
                best[sub] = melhor
            best[torch.sqrt(saa) < 1e-3 * (n1 * n1) ** 0.5] = -1.0  # recorte sem ruído (liso/saturado): não decide
            out.append(best.float().cpu().numpy())
    return np.concatenate(out)


def verified_models(gray_full: np.ndarray, gray: np.ndarray, s: float, cfg, k: int = 2, thr: float = 0.8, radius: int = 8,
                    min_pairs: int = 2, max_models: int = 8, kn: dict | None = None, sigma: float = 1.0) -> list[dict]:
    """Ligações sem razão (k vizinhos), cada uma confirmada nos pixels (correlação do ruído fino entre os dois recortes,
    no tamanho original; translação pura ou espelho). As confirmadas são agrupadas por deslocamento, do maior grupo
    para o menor; cada grupo vira um candidato de translação (formato de detector.seed_models, vseed = 1).
    gray = imagem na escala de trabalho (a que vai ao SIFT); s = escala de trabalho / original."""
    from .detector import verify_pairs
    m = links_from_knn(kn, cfg, k, gray.shape) if kn is not None else hough_matches(gray, cfg, k)
    if len(m["P"]) < min_pairs:
        return []
    P, Q, M = m["P"], m["Qo"], m["mir"]
    import time as _time

    from .detector import _cron
    t0 = _time.time()
    ncc = confirma_ligacoes(gray_full, P / s, Q / s, M, radius, sigma)
    _cron("ligacoes_confirma", t0)
    TEMPOS_N = len(P)
    ok = ncc >= thr
    P, Q, M, ncc = P[ok], Q[ok], M[ok], ncc[ok]
    from . import detector as _d
    _d.TEMPOS["n_ligacoes"] = _d.TEMPOS.get("n_ligacoes", 0.0) + TEMPOS_N
    _d.TEMPOS["n_confirmadas"] = _d.TEMPOS.get("n_confirmadas", 0.0) + len(P)
    if len(P) < min_pairs:
        return []
    t = Q - np.where(M[:, None], np.c_[-P[:, :1], P[:, 1:]], P)  # vetor do movimento
    tol = max(3.0 * cfg.ransac_thresh, 0.01 * np.hypot(*gray.shape))
    livre = np.ones(len(P), bool)
    rw = radius * s
    corners = np.array([[-rw, -rw], [rw, -rw], [rw, rw], [-rw, rw]], np.float32)
    out = []
    while len(out) < max_models and livre.sum() >= min_pairs:
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
