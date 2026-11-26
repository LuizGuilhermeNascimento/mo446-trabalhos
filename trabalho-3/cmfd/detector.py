"""Procura de cópias dentro da figura: SIFT casado com a própria imagem, RANSAC e máscara por correlação."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .knn import knn
from .ksift import sift_kornia


@dataclass
class Config:
    min_side_upscale: int           # imagens menores são ampliadas até este lado
    max_keypoints: int
    max_side: int = 2048
    clahe: float = 2.0
    knn: int = 8
    g2nn_ratio: float = 0.6
    min_dist_frac: float = 0.02     # pares mais próximos que isto (x diagonal) são ignorados
    min_dist_px: float = 12.0
    ransac_thresh: float = 3.0
    ransac_iters: int = 4000
    max_models: int = 8
    floor_inliers: int = 3
    min_scale: float = 0.25
    max_scale: float = 4.0
    max_aniso: float = 3.0
    zncc_win: int = 7
    zncc_win2: int = 21
    zncc_thr: float = 0.8
    flat_std: float = 2.0
    hull_margin_frac: float = 1.0
    min_comp_frac: float = 0.0002
    vlinks_k: int = 2               # ligações sem teste de razão, confirmadas pelo ruído (cmfd/ligacoes.py)
    vlinks_thr: float = 0.8
    vlinks_max: int = 8
    vlinks_radius: int = 4
    tile_ratio: float = 0.75        # ligações entre pedaços fixos (cmfd/tiles.py)
    tile_min_pairs: int = 4
    tile_max: int = 16


def prepare(gray: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    s = 1.0
    if max(h, w) > cfg.max_side:
        s = cfg.max_side / max(h, w)
    elif max(h, w) < cfg.min_side_upscale:
        s = min(4.0, cfg.min_side_upscale / max(h, w))
    if s != 1.0:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp)
    return gray, s


def self_matches(gray: np.ndarray, cfg: Config) -> dict:
    """Pares (P, Q) de pontos parecidos dentro da mesma imagem, incluindo a imagem espelhada."""
    h, w = gray.shape
    p0, d0 = sift_kornia(gray, cfg.max_keypoints)
    n0 = len(p0)
    pts, des, is_mirror = p0, d0, np.zeros(n0, bool)
    if n0:
        pm, dm = sift_kornia(cv2.flip(gray, 1), cfg.max_keypoints)
        if len(pm):
            pm = pm.copy()
            pm[:, 0] = (w - 1) - pm[:, 0]
            pts = np.vstack([p0, pm])
            des = np.vstack([d0, dm])
            is_mirror = np.r_[np.zeros(n0, bool), np.ones(len(pm), bool)]
    out = dict(P=np.zeros((0, 2), np.float32), Q=np.zeros((0, 2), np.float32), mirror=np.zeros(0, bool))
    if n0 < 3 or len(pts) < 4:
        return out
    I, Dd = knn(d0, des, min(cfg.knn + 1, len(pts)))
    out.update(kn=dict(p0=p0, pts=pts, I=I, n0=n0), desc=dict(p0=p0, d0=d0, pm=pts[n0:], dm=des[n0:]))
    min_d = max(cfg.min_dist_px, cfg.min_dist_frac * np.hypot(h, w))
    P, Q, M = [], [], []
    seen = set()
    for i in range(n0):
        row = [m for m in zip(I[i].tolist(), Dd[i].tolist()) if m[0] != i]
        for a in range(len(row) - 1):
            if row[a][1] >= cfg.g2nn_ratio * max(row[a + 1][1], 1e-6):
                break
            j = row[a][0]
            pi, pj = p0[i], pts[j]
            if np.hypot(*(pi - pj)) < min_d:
                continue
            key = (min(i, j), max(i, j), bool(is_mirror[j]))
            if key in seen:
                continue
            seen.add(key)
            P.append(pi); Q.append(pj); M.append(bool(is_mirror[j]))
    if P:
        out.update(P=np.float32(P), Q=np.float32(Q), mirror=np.array(M))
    return out


def valid_affine(A: np.ndarray, cfg: Config) -> tuple[bool, dict]:
    L = A[:, :2]
    sv = np.linalg.svd(L, compute_uv=False)
    det = float(np.linalg.det(L))
    ok = (cfg.min_scale <= sv.min() and sv.max() <= cfg.max_scale and sv.max() / max(sv.min(), 1e-9) <= cfg.max_aniso)
    return ok, dict(scale=float(np.sqrt(abs(det))), aniso=float(sv.max() / max(sv.min(), 1e-9)), reflect=det < 0)


def fit_models(P: np.ndarray, Q: np.ndarray, mirror: np.ndarray, cfg: Config) -> list[dict]:
    """Extrai movimentos afins por RANSAC até não sobrar suporte (cada par vale nos dois sentidos)."""
    models = []
    if len(P) < 3:
        return models
    alive = np.ones(len(P), bool)
    for _ in range(cfg.max_models):
        idx = np.flatnonzero(alive)
        if len(idx) < max(3, cfg.floor_inliers):
            break
        src = np.vstack([P[idx], Q[idx]])
        dst = np.vstack([Q[idx], P[idx]])
        A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh,
                                      maxIters=cfg.ransac_iters, confidence=0.995, refineIters=10)
        if A is None or inl is None:
            break
        inl = inl.ravel().astype(bool)
        fwd, bwd = inl[:len(idx)], inl[len(idx):]
        n_in = int(fwd.sum() + bwd.sum())
        if n_in < cfg.floor_inliers:
            break
        alive[idx[fwd | bwd]] = False
        ok, geo = valid_affine(A, cfg)
        S = np.vstack([P[idx][fwd], Q[idx][bwd]])
        T = np.vstack([Q[idx][fwd], P[idx][bwd]])
        shift = float(np.median(np.hypot(*(T - S).T))) if len(S) else 0.0
        spread = float(np.sqrt(np.prod(S.std(0) + 1e-6))) if len(S) > 1 else 0.0
        mir_frac = float(np.r_[mirror[idx][fwd], mirror[idx][bwd]].mean()) if n_in else 0.0
        models.append(dict(A=A, S=S, T=T, inliers=n_in, n_alive=int(len(idx)), inlier_ratio=n_in / max(len(idx), 1),
                           valid_geom=bool(ok), shift=shift, spread=spread, mirror_frac=mir_frac, **geo))
    return models


def detect_candidates(gray_full: np.ndarray, cfg: Config) -> list[dict]:
    """Movimentos candidatos de uma ampliação: RANSAC, pedaços fixos e ligações confirmadas pelo ruído."""
    from .ligacoes import verified_models
    from .tiles import tile_models
    gray, s = prepare(gray_full, cfg)
    gk = cv2.createCLAHE(clipLimit=cfg.clahe, tileGridSize=(8, 8)).apply(gray)
    m = self_matches(gk, cfg)
    models = fit_models(m["P"], m["Q"], m["mirror"], cfg)
    if "desc" in m:
        models += tile_models(m["desc"]["p0"], m["desc"]["d0"], m["desc"]["pm"], m["desc"]["dm"], gk.shape, cfg)
        models += verified_models(gray_full, gk.shape, s, cfg, m["kn"])
    cands = []
    for md in models:
        stats = {k: v for k, v in md.items() if k not in ("A", "S", "T")}
        stats["shift"] = stats["shift"] / s
        stats["spread"] = stats["spread"] / s
        cands.append(dict(stats=stats, A=np.asarray(md["A"], np.float32), S=np.asarray(md["S"], np.float32), T=np.asarray(md["T"], np.float32)))
    return cands


def hull_mask(pts: np.ndarray, shape, dilate: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2)).astype(np.int32)
        cv2.fillConvexPoly(m, hull, 1)
    else:
        for x, y in pts.astype(int):
            cv2.circle(m, (int(x), int(y)), max(dilate, 2), 1, -1)
    if dilate > 0:
        if dilate <= 8:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
        else:
            dist = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            m = (dist <= dilate).astype(np.uint8)
    return m


def fill_holes(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m).astype(bool)
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if len(rows) == 0:
        return m
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    c = np.ones((y1 - y0 + 2, x1 - x0 + 2), np.uint8)
    c[1:-1, 1:-1] = ~m[y0:y1, x0:x1]
    cv2.floodFill(c, None, (0, 0), 2, flags=4)
    out = m.copy()
    out[y0:y1, x0:x1] |= c[1:-1, 1:-1] == 1
    return out


def _local_stats(a: np.ndarray, win: int):
    k = (win, win)
    mu = cv2.boxFilter(a, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.boxFilter(a * a, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    return mu, np.maximum(mu2 - mu * mu, 0)


def _zncc(G: np.ndarray, J: np.ndarray, valid: np.ndarray, win: int, flat_std: float) -> np.ndarray:
    mg, vg = _local_stats(G, win)
    mj, vj = _local_stats(J, win)
    cov = cv2.boxFilter(G * J, cv2.CV_32F, (win, win), borderType=cv2.BORDER_REFLECT) - mg * mj
    sg, sj = np.sqrt(vg), np.sqrt(vj)
    textured = (sg > flat_std) & (sj > flat_std) & valid
    return np.where(textured, cov / (sg * sj + 1e-6), 0)


def dense_masks(gray: np.ndarray, A: np.ndarray, S: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras (origem, cópia): ZNCC local entre I(x) e I(A x), em volta dos pontos de origem S."""
    h, w = gray.shape
    G = gray.astype(np.float32)
    J = cv2.warpAffine(G, A, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), A, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    J = np.where(valid, J, 0).astype(np.float32)
    side = np.sqrt(max(cv2.contourArea(cv2.convexHull(S.reshape(-1, 1, 2))) if len(S) >= 3 else 0, 1.0))
    margin = int(max(4 * cfg.zncc_win, cfg.hull_margin_frac * side))
    kc = max(3, cfg.zncc_win) | 1
    pad = max(cfg.zncc_win, cfg.zncc_win2) // 2 + 2 * kc + 8 + max(margin, 2)
    x0 = max(int(np.floor(S[:, 0].min())) - pad, 0)
    y0 = max(int(np.floor(S[:, 1].min())) - pad, 0)
    x1 = min(int(np.ceil(S[:, 0].max())) + pad + 1, w)
    y1 = min(int(np.ceil(S[:, 1].max())) + pad + 1, h)
    Gc = np.ascontiguousarray(G[y0:y1, x0:x1])
    Jc = np.ascontiguousarray(J[y0:y1, x0:x1])
    vc = np.ascontiguousarray(valid[y0:y1, x0:x1])
    cand = _zncc(Gc, Jc, vc, cfg.zncc_win, cfg.flat_std) > cfg.zncc_thr
    cand |= _zncc(Gc, Jc, vc, cfg.zncc_win2, cfg.flat_std) > cfg.zncc_thr
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand &= hull_mask((S - np.array([x0, y0], S.dtype)).astype(S.dtype), cand.shape, margin)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    min_px = max(cfg.zncc_win * cfg.zncc_win, int(cfg.min_comp_frac * h * w))
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
    if not keep:
        src = hull_mask(S, (h, w), cfg.zncc_win // 2)
    else:
        sc = np.isin(lab, keep).astype(np.uint8)
        sc = cv2.morphologyEx(sc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kc + 1, 2 * kc + 1)))
        src = np.zeros((h, w), np.uint8)
        src[y0:y1, x0:x1] = fill_holes(sc)
    dst = cv2.warpAffine(src, A, (w, h), flags=cv2.INTER_NEAREST)
    return src, dst


def noise_residual(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    G = gray.astype(np.float32)
    return G - cv2.GaussianBlur(G, (0, 0), sigma)


def residual_ncc(r: np.ndarray, A: np.ndarray, src: np.ndarray, search: float = 1.0, max_pts: int = 20000):
    """Correlação do ruído r entre a origem (máscara src) e a cópia (A x): (melhor valor, deslocamento subpixel)."""
    ys, xs = np.nonzero(src)
    if len(ys) < 16:
        return 0.0, (0.0, 0.0)
    if len(ys) > max_pts:
        sel = np.random.default_rng(0).choice(len(ys), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    a = r[ys, xs].astype(np.float64)
    a -= a.mean()
    na = np.sqrt((a * a).sum())
    if na < 1e-3:
        return 0.0, (0.0, 0.0)
    X = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    Y = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    best, off = -1.0, (0.0, 0.0)
    steps = np.arange(-search, search + 1e-6, 0.5)
    for dy in steps:
        for dx in steps:
            mx = (X + dx).astype(np.float32).reshape(1, -1)
            my = (Y + dy).astype(np.float32).reshape(1, -1)
            b = cv2.remap(r, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()
            b = b.astype(np.float64) - b.mean()
            nb = np.sqrt((b * b).sum())
            if nb > 1e-3:
                v = float((a * b).sum() / (na * nb))
                if v > best:
                    best, off = v, (float(dx), float(dy))
    return best, off


def noise_refine(r: np.ndarray, A: np.ndarray, src: np.ndarray, pre: tuple, win: int = 5, thr: float = 0.5, low: float = 0.2,
                 min_std: float = 0.5, grow_frac: float = 0.5, iters: int = 3, gate: float = 0.3) -> np.ndarray:
    """Refina a máscara de origem pelo mapa local de correlação do ruído r entre x e A x; pre = residual_ncc."""
    H, W = r.shape
    src = src.astype(bool)
    if src.sum() < 16:
        return src
    rn, (dx, dy) = pre
    if rn < gate:
        return src
    rad = int(max(win, grow_frac * np.sqrt(src.sum())))
    ys, xs = np.nonzero(src)
    pad = rad * iters + win + 8
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, H)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, W)
    h, w = y1 - y0, x1 - x0
    A2 = A.astype(np.float64).copy()
    A2[0, 2] += dx
    A2[1, 2] += dy
    A2[:, 2] += A2[:, :2] @ np.array([x0, y0], np.float64)
    rc = np.ascontiguousarray(r[y0:y1, x0:x1])
    J = cv2.warpAffine(r, A2, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((H, W), np.uint8), A2, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    k = (win, win)
    mr, mj = cv2.blur(rc, k), cv2.blur(J, k)
    vr = cv2.blur(rc * rc, k) - mr * mr
    vj = cv2.blur(J * J, k) - mj * mj
    cov = cv2.blur(rc * J, k) - mr * mj
    known = (vr > min_std ** 2) & (vj > min_std ** 2) & valid
    ncc = cov / np.sqrt(np.maximum(vr * vj, 1e-6))
    good, bad = known & (ncc > thr), known & (ncc < low)
    s0 = src[y0:y1, x0:x1]
    cur = s0
    for _ in range(iters):
        region = cv2.distanceTransform((~cur).astype(np.uint8), cv2.DIST_L2, 5) <= rad
        new = (cur & ~bad) | (good & region)
        new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(new, connectivity=8)
        if n <= 1:
            return src
        touch = np.unique(lab[s0 & (lab > 0)])
        new = np.isin(lab, touch)
        new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        new = fill_holes(new)
        if new.sum() < 0.3 * s0.sum():
            return src
        if (new == cur).all():
            break
        cur = new
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = cur
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


def compose(cands: list, idxs, dedup_iou: float = 0.7, merge_frac: float = 0.3, max_masks: int = 4) -> list[np.ndarray]:
    """Máscaras finais dos candidatos escolhidos (em ordem): tira repetidos, junta os que se sobrepõem."""
    parts = []
    for i in idxs:
        p = cands[i]["src"] | cands[i]["dst"]
        if not parts or max(_iou(p, q) for q in parts) <= dedup_iou:
            parts.append(p)
    n = len(parts)
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    area = [int(p.sum()) for p in parts]
    for a in range(n):
        for b in range(a + 1, n):
            if find(a) != find(b):
                inter = int(np.logical_and(parts[a], parts[b]).sum())
                if inter > 0 and inter > merge_frac * min(area[a], area[b]):
                    parent[find(a)] = find(b)
    groups: dict[int, np.ndarray] = {}
    for k in range(n):
        r = find(k)
        groups[r] = groups[r] | parts[k] if r in groups else parts[k].copy()
    return [m for m in groups.values() if m.any()][:max_masks]
