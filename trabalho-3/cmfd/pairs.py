"""Cópia entre fotos da mesma figura: cada par de fotos (caixas do SAM) é comparado diretamente."""
from __future__ import annotations

import cv2
import numpy as np

from .knn import knn
from .peak import peak_sharpness

MIN_LADO, MAX_FRAC, MAX_IOU = 64, 0.8, 0.5      # caixa pequena, gigante ou repetida não é uma foto
LADO_MAX = 1024                                 # foto maior é reduzida até este lado
RAZAO, RANSAC_PX, ESCALA = 0.75, 3.0, (0.5, 2.0)
MAX_DENTRO = 0.2                                # par pulado se uma caixa cobre mais que isto da outra
MIN_PONTOS, REL = 20, 0.2
MIN_DESL_PX, MIN_DESL_FRAC = 12.0, 0.02


def _sift(gray: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.002, edgeThreshold=10.0)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    des = des.astype(np.float32)
    des /= des.sum(1, keepdims=True) + 1e-7
    return np.float32([k.pt for k in kps]), np.sqrt(des)


def _inter(c, k):
    return max(0, min(c[2], k[2]) - max(c[0], k[0])) * max(0, min(c[3], k[3]) - max(c[1], k[1]))


def _area(c):
    return (c[2] - c[0]) * (c[3] - c[1])


def caixas(B: np.ndarray, H: int, W: int) -> list[tuple]:
    """Caixas do SAM (frações da figura) -> caixas em pixels, sem as pequenas, as gigantes e as repetidas."""
    out = []
    for x0, y0, x1, y1 in B:
        c = (int(round(x0 * W)), int(round(y0 * H)), min(W, int(round(x1 * W))), min(H, int(round(y1 * H))))
        if min(c[2] - c[0], c[3] - c[1]) < MIN_LADO or _area(c) > MAX_FRAC * H * W:
            continue
        out.append(c)
    out.sort(key=lambda c: -_area(c))
    keep = []
    for c in out:
        if all(_inter(c, k) / (_area(c) + _area(k) - _inter(c, k)) <= MAX_IOU for k in keep):
            keep.append(c)
    return keep


def _descreve(g: np.ndarray, c: tuple) -> dict:
    rec = g[c[1]:c[3], c[0]:c[2]]
    s = 1.0
    if max(rec.shape) > LADO_MAX:
        s = LADO_MAX / max(rec.shape)
        rec = cv2.resize(rec, (max(1, round(rec.shape[1] * s)), max(1, round(rec.shape[0] * s))), interpolation=cv2.INTER_AREA)
    rec = np.ascontiguousarray(rec)
    p, d = _sift(rec)
    pf, df = _sift(cv2.flip(rec, 1))
    return dict(c=c, s=s, wh=(rec.shape[1], rec.shape[0]), p=p, d=d, pf=pf, df=df)


def _casa(A: dict, B: dict, flip: bool):
    """Similaridade A -> B: (nº de pontos concordantes distintos, M 2x3, pontos concordantes em A) ou (0, None, None)."""
    pb, db = (B["pf"], B["df"]) if flip else (B["p"], B["d"])
    if len(A["p"]) < 3 or len(pb) < 3:
        return 0, None, None
    I, D = knn(A["d"], db, 2)
    ok = D[:, 0] < RAZAO * D[:, 1]
    if ok.sum() < 3:
        return 0, None, None
    S, T = A["p"][ok], pb[I[ok, 0]]
    M, inl = cv2.estimateAffinePartial2D(S, T, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX, maxIters=2000, confidence=0.999)
    if M is None:
        return 0, None, None
    esc = float(np.hypot(M[0, 0], M[1, 0])) * A["s"] / B["s"]
    if not (ESCALA[0] <= esc <= ESCALA[1]):
        return 0, None, None
    k = inl.ravel().astype(bool)
    return min(len(np.unique(np.round(S[k]), axis=0)), len(np.unique(np.round(T[k]), axis=0))), M, S[k]


def _para_figura(P: dict, pts, flip: bool):
    pts = np.asarray(pts, np.float64).copy()
    if flip:
        pts[:, 0] = (P["wh"][0] - 1) - pts[:, 0]
    return pts / P["s"] + np.array(P["c"][:2], np.float64)


def _poligono(shape, pts, c):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
    dentro = np.zeros(shape, bool)
    dentro[c[1]:c[3], c[0]:c[2]] = True
    return m.astype(bool) & dentro


def _ret(P: dict):
    w, h = P["wh"]
    return np.float64([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def _regiao(shape, A: dict, B: dict, flip: bool, M) -> np.ndarray:
    """A área em que uma foto cai sobre a outra pelo movimento, nas duas fotos."""
    a_em_b = cv2.transform(_ret(A)[None], M)[0]
    b_em_a = cv2.transform(_ret(B)[None], cv2.invertAffineTransform(M))[0]
    return _poligono(shape, _para_figura(B, a_em_b, flip), B["c"]) | _poligono(shape, _para_figura(A, b_em_a, False), A["c"])


def movimento(A: dict, B: dict, flip: bool, M) -> np.ndarray:
    """Movimento da foto A para a foto B em pixels da figura (3x3)."""
    T1 = np.array([[A["s"], 0, -A["c"][0] * A["s"]], [0, A["s"], -A["c"][1] * A["s"]], [0, 0, 1.0]])
    F = np.array([[-1.0, 0, B["wh"][0] - 1], [0, 1, 0], [0, 0, 1]]) if flip else np.eye(3)
    T2 = np.array([[1 / B["s"], 0, B["c"][0]], [0, 1 / B["s"], B["c"][1]], [0, 0, 1.0]])
    return T2 @ F @ np.vstack([M, [0, 0, 1.0]]) @ T1


def confere(gray: np.ndarray, gf: np.ndarray, A: dict, B: dict, flip: bool, M) -> tuple[float, float]:
    """(deslocamento mediano em px da região de A sob o movimento do par, correlação do conteúdo dessa região com B)."""
    Af = movimento(A, B, flip, M)
    reg = cv2.transform(_ret(B)[None], cv2.invertAffineTransform(M))[0]
    src = _poligono(gray.shape, _para_figura(A, reg, False), A["c"])
    ys, xs = np.nonzero(src)
    if not len(ys):
        return 0.0, 0.0
    q = np.c_[xs, ys].astype(np.float64)[:: max(1, len(ys) // 2000)]
    desl = float(np.median(np.hypot(*((q @ Af[:2, :2].T + Af[:2, 2]) - q).T)))
    sub = src
    if len(ys) > 40000:
        sub = np.zeros_like(src)
        st = int(np.ceil(np.sqrt(len(ys) / 40000)))
        sub[::st, ::st] = src[::st, ::st]
    return desl, peak_sharpness(gf, Af[:2], sub, W=2, guard=2)["pico"]


def candidatos(gray: np.ndarray, B: np.ndarray) -> tuple[list, list]:
    """(fotos descritas, pares com >= MIN_PONTOS como (n, i, j, espelhado, M), do mais forte ao mais fraco)."""
    H, W = gray.shape
    cx = caixas(B, H, W)
    if len(cx) < 2:
        return [], []
    ps = [_descreve(gray, c) for c in cx]
    ach = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if _inter(cx[i], cx[j]) > MAX_DENTRO * min(_area(cx[i]), _area(cx[j])):
                continue
            n, M, f = 0, None, False
            for flip in (False, True):
                n2, M2, _ = _casa(ps[i], ps[j], flip)
                if n2 > n:
                    n, M, f = n2, M2, flip
            if n >= MIN_PONTOS:
                ach.append((n, i, j, f, M))
    ach.sort(key=lambda x: -x[0])
    return ps, ach


def monta(regs: list) -> list:
    """[(i, j, máscara do par)] -> máscaras finais: pares que dividem uma foto viram uma só; vazia ou já coberta sai."""
    pai = {}

    def raiz(u):
        while pai.setdefault(u, u) != u:
            u = pai[u]
        return u
    for i, j, _ in regs:
        pai[raiz(i)] = raiz(j)
    grupos = {}
    for i, j, m in regs:
        grupos.setdefault(raiz(i), []).append(m)
    out, cob = [], None
    for m in (np.any(v, 0) for v in grupos.values()):
        if not m.any() or (cob is not None and (m & cob).sum() >= 0.5 * m.sum()):
            continue
        out.append(m)
        cob = m.copy() if cob is None else (cob | m)
    return out


def pair_masks(gray: np.ndarray, B: np.ndarray) -> list:
    """Figura em tons de cinza + caixas do SAM -> máscaras dos pares de fotos que casam ([] = nenhum)."""
    ps, ach = candidatos(gray, B)
    if not ach:
        return []
    gf = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    min_d = max(MIN_DESL_PX, MIN_DESL_FRAC * np.hypot(*gray.shape))
    fica = []
    for x in ach:
        desl, cz = confere(gray, gf, ps[x[1]], ps[x[2]], x[3], x[4])
        if not (desl < min_d or cz < 0.0):
            fica.append(x)
    if not fica:
        return []
    regs = [(x[0], x[1], x[2], _regiao(gray.shape, ps[x[1]], ps[x[2]], x[3], x[4])) for x in fica if x[0] >= REL * fica[0][0]]
    return monta([r[1:] for r in regs])
