"""Candidatos de copy-move por campo denso de correspondências (sem treino).

Segue Cozzolino, Poggi e Verdoliva, "Efficient dense-field copy-move forgery
detection", IEEE TIFS 10(11), 2015:

1. cada pixel é descrito pelos módulos dos momentos de Zernike do patch circular em
   volta dele (invariantes a giro e a espelho);
2. PatchMatch acha, para cada pixel, o pixel mais parecido fora de uma distância
   mínima (campo de deslocamentos);
3. numa cópia rígida o campo é localmente LINEAR (translação: constante; giro/zoom:
   varia linearmente); fora dela é ruído. O erro do ajuste linear local (DLF) separa
   os dois;
4. componentes coerentes viram movimentos afins (mínimos quadrados) que seguem para a
   mesma máscara densa (ZNCC) e os mesmos filtros do detector por SIFT.

Funciona pixel a pixel, por isso alcança regiões pequenas e lisas onde não há
pontos-chave suficientes para o RANSAC.
"""
from __future__ import annotations

import cv2
import numpy as np
from numba import njit

# ordens (n, m) dos momentos de Zernike até n = 5 (12 módulos)
ZM_ORDERS = [(n, m) for n in range(6) for m in range(n + 1) if (n - m) % 2 == 0]


def _zernike_kernels(radius: int):
    r = np.arange(-radius, radius + 1, dtype=np.float64)
    X, Y = np.meshgrid(r, r)
    rho = np.hypot(X, Y) / (radius + 0.5)
    th = np.arctan2(Y, X)
    inside = rho <= 1.0
    ks = []
    from math import factorial
    for n, m in ZM_ORDERS:
        R = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            c = (-1) ** k * factorial(n - k) / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
            R += c * rho ** (n - 2 * k)
        V = R * np.exp(-1j * m * th) * inside * (n + 1) / np.pi
        ks.append((V.real.astype(np.float32), V.imag.astype(np.float32)))
    return ks, inside


def zernike_features(gray: np.ndarray, radius: int = 8) -> np.ndarray:
    """(h, w, 12) módulos dos momentos de Zernike do disco de raio `radius` em cada pixel."""
    G = gray.astype(np.float32)
    ks, _ = _zernike_kernels(radius)
    feats = []
    for kr, ki in ks:
        re = cv2.filter2D(G, cv2.CV_32F, kr, borderType=cv2.BORDER_REFLECT)
        im = cv2.filter2D(G, cv2.CV_32F, ki, borderType=cv2.BORDER_REFLECT)
        feats.append(np.sqrt(re * re + im * im))
    return np.ascontiguousarray(np.stack(feats, -1))


@njit(cache=True)
def _dist(F, y, x, ty, tx, best):
    d = 0.0
    for k in range(F.shape[2]):
        t = F[y, x, k] - F[ty, tx, k]
        d += t * t
        if d >= best:
            return d
    return d


@njit(cache=True)
def _ok(y, x, ty, tx, h, w, dmin2):
    if ty < 0 or ty >= h or tx < 0 or tx >= w:
        return False
    dy, dx = ty - y, tx - x
    return dy * dy + dx * dx >= dmin2


@njit(cache=True)
def patchmatch(F, dmin, n_iter, seed):
    """Campo de vizinho mais próximo (oy, ox) com |deslocamento| >= dmin; devolve também a distância."""
    np.random.seed(seed)
    h, w = F.shape[0], F.shape[1]
    oy = np.zeros((h, w), np.int32)
    ox = np.zeros((h, w), np.int32)
    D = np.empty((h, w), np.float32)
    dmin2 = dmin * dmin
    for y in range(h):
        for x in range(w):
            while True:
                ty, tx = np.random.randint(0, h), np.random.randint(0, w)
                if _ok(y, x, ty, tx, h, w, dmin2):
                    break
            oy[y, x], ox[y, x] = ty - y, tx - x
            D[y, x] = _dist(F, y, x, ty, tx, np.inf)
    rmax = max(h, w)
    for it in range(n_iter):
        if it % 2 == 0:
            y0, y1, st = 0, h, 1
        else:
            y0, y1, st = h - 1, -1, -1
        y = y0
        while y != y1:
            x = 0 if st == 1 else w - 1
            for _ in range(w):
                best = D[y, x]
                # propagação: deslocamento do vizinho já visitado (esquerda/cima ou direita/baixo)
                for ny, nx in ((y, x - st), (y - st, x)):
                    if 0 <= ny < h and 0 <= nx < w:
                        ty, tx = y + oy[ny, nx], x + ox[ny, nx]
                        if _ok(y, x, ty, tx, h, w, dmin2):
                            d = _dist(F, y, x, ty, tx, best)
                            if d < best:
                                best = d
                                oy[y, x], ox[y, x] = ty - y, tx - x
                # busca aleatória em raios decrescentes
                r = rmax
                while r >= 1:
                    ty = y + oy[y, x] + np.random.randint(-r, r + 1)
                    tx = x + ox[y, x] + np.random.randint(-r, r + 1)
                    if _ok(y, x, ty, tx, h, w, dmin2):
                        d = _dist(F, y, x, ty, tx, best)
                        if d < best:
                            best = d
                            oy[y, x], ox[y, x] = ty - y, tx - x
                    r //= 2
                D[y, x] = best
                x += st
            y += st
    return oy, ox, D


def dlf_error(oy: np.ndarray, ox: np.ndarray, radius: int) -> np.ndarray:
    """Erro médio (px²) do ajuste linear local f(u, v) = a + b u + c v aos dois campos."""
    k = 2 * radius + 1
    r = np.arange(-radius, radius + 1, dtype=np.float32)
    U = np.tile(r, (k, 1))
    V = U.T.copy()
    su2 = float((U * U).sum())
    n = float(k * k)
    err = np.zeros(oy.shape, np.float32)
    for f in (oy.astype(np.float32), ox.astype(np.float32)):
        s1 = cv2.boxFilter(f, cv2.CV_32F, (k, k), normalize=False, borderType=cv2.BORDER_REFLECT)
        s2 = cv2.boxFilter(f * f, cv2.CV_32F, (k, k), normalize=False, borderType=cv2.BORDER_REFLECT)
        su = cv2.filter2D(f, cv2.CV_32F, U, borderType=cv2.BORDER_REFLECT)
        sv = cv2.filter2D(f, cv2.CV_32F, V, borderType=cv2.BORDER_REFLECT)
        err += s2 - s1 * s1 / n - su * su / su2 - sv * sv / su2
    return np.maximum(err, 0) / n


def field_models(gray: np.ndarray, max_side: int = 768, radius: int = 8, dmin_frac: float = 0.03,
                 n_iter: int = 6, dlf_radius: int = 4, dlf_thr: float = 1.0, flat_std: float = 3.0,
                 min_area_frac: float = 2e-4, min_area_px: int = 48, max_models: int = 4, seed: int = 0):
    """Movimentos afins (na escala de `gray`) a partir do campo denso coerente.

    Devolve lista de dicts no formato de detector.fit_models (A, S, T, estatísticas) e
    a escala usada. `inliers` = nº de pixels coerentes (na escala do campo).
    """
    h0, w0 = gray.shape
    s = min(1.0, max_side / max(h0, w0))
    g = cv2.resize(gray, (max(1, int(round(w0 * s))), max(1, int(round(h0 * s)))), interpolation=cv2.INTER_AREA) \
        if s < 1 else gray
    h, w = g.shape
    F = zernike_features(g, radius)
    dmin = max(2 * radius + 1, int(dmin_frac * np.hypot(h, w)))
    oy, ox, _ = patchmatch(F, dmin, n_iter, seed)
    # mediana 5x5 tira deslocamentos isolados antes do ajuste local
    oyf = cv2.medianBlur(oy.astype(np.float32), 5)
    oxf = cv2.medianBlur(ox.astype(np.float32), 5)
    err = dlf_error(oyf, oxf, dlf_radius)
    G = g.astype(np.float32)
    mu = cv2.boxFilter(G, cv2.CV_32F, (2 * radius + 1, 2 * radius + 1))
    sd = np.sqrt(np.maximum(cv2.boxFilter(G * G, cv2.CV_32F, (2 * radius + 1, 2 * radius + 1)) - mu * mu, 0))
    mag = np.hypot(oyf, oxf)
    good = ((err < dlf_thr) & (sd > flat_std) & (mag >= dmin)).astype(np.uint8)
    good = cv2.morphologyEx(good, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(good, connectivity=8)
    min_px = max(min_area_px, int(min_area_frac * h * w))
    comps = sorted((i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px),
                   key=lambda i: -st[i, cv2.CC_STAT_AREA])
    used = np.zeros(n, bool)
    models = []
    for i in comps:
        if used[i] or len(models) >= max_models:
            continue
        ys, xs = np.nonzero(lab == i)
        P = np.stack([xs, ys], 1).astype(np.float32)
        Q = np.stack([xs + oxf[ys, xs], ys + oyf[ys, xs]], 1).astype(np.float32)
        A, inl = cv2.estimateAffine2D(P, Q, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=500,
                                      confidence=0.99, refineIters=10)
        if A is None:
            continue
        inl = inl.ravel().astype(bool)
        # o par simétrico (a cópia apontando de volta para a origem) é o mesmo movimento
        qx = np.clip(np.round(Q[inl, 0]).astype(int), 0, w - 1)
        qy = np.clip(np.round(Q[inl, 1]).astype(int), 0, h - 1)
        hit = np.bincount(lab[qy, qx], minlength=n)
        hit[0] = 0
        used[i] = True
        used[hit > 0.3 * inl.sum()] = True
        L = A[:, :2]
        sv = np.linalg.svd(L, compute_uv=False)
        det = float(np.linalg.det(L))
        Sx, Tx = P[inl] / s, Q[inl] / s
        A2 = A.astype(np.float64).copy()
        A2[:, 2] /= s
        # subamostra os pontos (bastam para casco/ECC; o cache fica leve)
        step = max(1, len(Sx) // 400)
        models.append(dict(A=A2, S=Sx[::step], T=Tx[::step], inliers=int(inl.sum()), n_alive=int(len(P)),
                           inlier_ratio=float(inl.mean()), valid_geom=True,
                           shift=float(np.median(np.hypot(*(Tx - Sx).T))),
                           spread=float(np.sqrt(np.prod(Sx.std(0) + 1e-6))), mirror_frac=float(det < 0),
                           scale=float(np.sqrt(abs(det))), aniso=float(sv.max() / max(sv.min(), 1e-9)),
                           reflect=det < 0, support_frac=float(inl.sum()) / (h * w), field_dlf=float(err[ys, xs].mean())))
    return models, s
