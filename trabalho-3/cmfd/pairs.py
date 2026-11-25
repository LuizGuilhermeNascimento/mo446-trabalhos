"""Cópia ENTRE fotos da mesma figura: cada par de painéis é comparado diretamente (como os primeiros colocados).

As caixas dos painéis vêm do SAM 3.1 congelado (kaggle/sam_paineis.py, campo B). Cada painel é recortado, recebe
SIFT próprio e é casado com cada um dos outros; uma similaridade (deslocamento, giro, escala) por RANSAC dá o número
de pontos concordantes do par. Ficam os pares fortes (>= min_pontos e >= rel x o mais forte); pares que dividem um
painel viram uma máscara só; a máscara é a área em que um painel cai sobre o outro.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .detector import _sift
from .knn import knn
from .variants import VARIANTS


@dataclass
class PairConfig:
    min_lado: int = 64          # caixa com lado menor que isto não entra
    max_frac: float = 0.8       # caixa maior que esta fração da figura não é um painel
    max_iou: float = 0.5        # caixas que coincidem mais que isto são o mesmo painel
    lado_min: int = 512         # painel menor é ampliado até este lado (0 = não amplia)
    lado_max: int = 1024        # painel maior é reduzido até este lado (0 = não reduz)
    amplia_max: float = 4.0
    razao: float = 0.75         # 1º vizinho / 2º vizinho
    ransac_px: float = 3.0
    escala: tuple = (0.5, 2.0)  # escala aceita entre os dois painéis, em pixels da figura
    max_dentro: float = 0.2     # par pulado se uma caixa cobre mais que isto da outra
    espelho: bool = True        # testa também o painel virado na horizontal
    min_pontos: int = 20
    min_area: float = 0.0       # área copiada (nas duas fotos) >= esta fração de 2x a menor foto; 0 = sem corte
    rel: float = 0.2
    agrupa: bool = True
    min_paineis: int = 2        # figura com menos painéis que isto não usa os pares
    sem_auto: bool = False      # um trecho não é cópia de si mesmo: par cujo movimento não desloca a região sai (caixas sobrepostas)
    min_desl_px: float = 12.0   # deslocamento mínimo, como no método (detector.Config.min_dist_px / min_dist_frac)
    min_desl_frac: float = 0.02
    cz_casco: bool = False      # True = correlação medida só no casco dos pontos concordantes (testado: não separa os alarmes)
    min_cz: float = 0.0         # correlação mínima do conteúdo da região da foto A com a foto B pelo movimento do par; 0 = sem corte
    sift: str = "f3_1024"       # ajustes do SIFT (cmfd.variants)
    sift_impl: str = "opencv"   # a comparação entre fotos usa sempre o SIFT do OpenCV (o do Kornia piora as várias fotos no treino)


def _inter(c, k):
    return max(0, min(c[2], k[2]) - max(c[0], k[0])) * max(0, min(c[3], k[3]) - max(c[1], k[1]))


def _area(c):
    return (c[2] - c[0]) * (c[3] - c[1])


def caixas(B: np.ndarray, H: int, W: int, cfg: PairConfig) -> list[tuple]:
    """Caixas do SAM (frações da figura) -> caixas em pixels, sem as pequenas, as gigantes e as repetidas."""
    out = []
    for x0, y0, x1, y1 in B:
        c = (int(round(x0 * W)), int(round(y0 * H)), min(W, int(round(x1 * W))), min(H, int(round(y1 * H))))
        if min(c[2] - c[0], c[3] - c[1]) < cfg.min_lado or _area(c) > cfg.max_frac * H * W:
            continue
        out.append(c)
    out.sort(key=lambda c: -_area(c))
    keep = []
    for c in out:
        if all(_inter(c, k) / (_area(c) + _area(k) - _inter(c, k)) <= cfg.max_iou for k in keep):
            keep.append(c)
    return keep


def _descreve(g: np.ndarray, c: tuple, cfg: PairConfig) -> dict:
    rec = g[c[1]:c[3], c[0]:c[2]]
    lado = max(rec.shape)
    s = 1.0
    if cfg.lado_min and lado < cfg.lado_min:
        s = min(cfg.amplia_max, cfg.lado_min / lado)
    elif cfg.lado_max and lado > cfg.lado_max:
        s = cfg.lado_max / lado
    if s != 1.0:
        rec = cv2.resize(rec, (max(1, round(rec.shape[1] * s)), max(1, round(rec.shape[0] * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rec = np.ascontiguousarray(rec)
    sc = replace(VARIANTS[cfg.sift], sift_impl=cfg.sift_impl)
    p, d = _sift(rec, sc)
    pf, df = _sift(cv2.flip(rec, 1), sc) if cfg.espelho else (None, None)
    return dict(c=c, s=s, wh=(rec.shape[1], rec.shape[0]), p=p, d=d, pf=pf, df=df)


def _casa(A: dict, B: dict, flip: bool, cfg: PairConfig):
    """Similaridade A -> B. Devolve (nº de pontos concordantes distintos, M 2x3, pontos concordantes em A) ou (0, None, None)."""
    pb, db = (B["pf"], B["df"]) if flip else (B["p"], B["d"])
    if len(A["p"]) < 3 or len(pb) < 3:
        return 0, None, None
    I, D = knn(A["d"], db, 2)  # exato: a mesma resposta do cv2.BFMatcher
    ok = D[:, 0] < cfg.razao * D[:, 1]
    if ok.sum() < 3:
        return 0, None, None
    S, T = A["p"][ok], pb[I[ok, 0]]
    M, inl = cv2.estimateAffinePartial2D(S, T, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_px, maxIters=2000, confidence=0.999)
    if M is None:
        return 0, None, None
    esc = float(np.hypot(M[0, 0], M[1, 0])) * A["s"] / B["s"]
    if not (cfg.escala[0] <= esc <= cfg.escala[1]):
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
    """A área em que um painel cai sobre o outro pela transformação, nos dois painéis."""
    a_em_b = cv2.transform(_ret(A)[None], M)[0]
    b_em_a = cv2.transform(_ret(B)[None], cv2.invertAffineTransform(M))[0]
    return _poligono(shape, _para_figura(B, a_em_b, flip), B["c"]) | _poligono(shape, _para_figura(A, b_em_a, False), A["c"])


def movimento(A: dict, B: dict, flip: bool, M) -> np.ndarray:
    """Movimento da foto A para a foto B em pixels da figura (3x3)."""
    T1 = np.array([[A["s"], 0, -A["c"][0] * A["s"]], [0, A["s"], -A["c"][1] * A["s"]], [0, 0, 1.0]])
    F = np.array([[-1.0, 0, B["wh"][0] - 1], [0, 1, 0], [0, 0, 1]]) if flip else np.eye(3)
    T2 = np.array([[1 / B["s"], 0, B["c"][0]], [0, 1 / B["s"], B["c"][1]], [0, 0, 1.0]])
    return T2 @ F @ np.vstack([M, [0, 0, 1.0]]) @ T1


def confere(gray: np.ndarray, gf: np.ndarray, A: dict, B: dict, flip: bool, M, Sk=None) -> tuple[float, float]:
    """(deslocamento mediano em px da região de A sob o movimento do par, correlação do conteúdo dessa região com B).
    A região é o casco dos pontos concordantes em A (Sk, onde a cópia foi de fato vista); sem Sk, toda a área em que
    uma foto cai sobre a outra. gf = figura borrada (sigma 1) em float32."""
    from .peak import peak_sharpness
    Af = movimento(A, B, flip, M)
    reg = cv2.transform(_ret(B)[None], cv2.invertAffineTransform(M))[0] if Sk is None else np.asarray(Sk, np.float64)
    src = _poligono(gray.shape, _para_figura(A, reg, False), A["c"])
    ys, xs = np.nonzero(src)
    if not len(ys):
        return 0.0, 0.0
    q = np.c_[xs, ys].astype(np.float64)[:: max(1, len(ys) // 2000)]
    desl = float(np.median(np.hypot(*((q @ Af[:2, :2].T + Af[:2, 2]) - q).T)))
    sub = src
    if len(ys) > 40000:  # região grande: grade regular de pixels
        sub = np.zeros_like(src)
        st = int(np.ceil(np.sqrt(len(ys) / 40000)))
        sub[::st, ::st] = src[::st, ::st]
    return desl, peak_sharpness(gf, Af[:2], sub, W=2, guard=2)["pico"]


def candidatos(gray: np.ndarray, B: np.ndarray, cfg: PairConfig = PairConfig()) -> tuple[list, list, list]:
    """(caixas, painéis descritos, pares com >= min_pontos como (n, i, j, espelhado, M), do mais forte ao mais fraco)."""
    H, W = gray.shape
    cx = caixas(B, H, W, cfg)
    if len(cx) < max(2, cfg.min_paineis):
        return cx, [], []
    ps = [_descreve(gray, c, cfg) for c in cx]
    ach = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if _inter(cx[i], cx[j]) > cfg.max_dentro * min(_area(cx[i]), _area(cx[j])):
                continue
            n, M, f, Sk = 0, None, False, None
            for flip in ((False, True) if cfg.espelho else (False,)):
                n2, M2, S2 = _casa(ps[i], ps[j], flip, cfg)
                if n2 > n:  # empate fica com o não espelhado
                    n, M, f, Sk = n2, M2, flip, S2
            if n >= cfg.min_pontos:
                ach.append((n, i, j, f, M, Sk))  # Sk: pontos concordantes na foto i (coordenadas do recorte)
    ach.sort(key=lambda x: -x[0])  # estável: empate fica na ordem (i, j)
    return cx, ps, ach


def monta(regs: list, agrupa: bool = True) -> list:
    """[(i, j, máscara do par)] -> máscaras finais: pares que dividem um painel viram uma só; vazia ou já coberta sai."""
    if agrupa:
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
        ms = [np.any(v, 0) for v in grupos.values()]
    else:
        ms = [m for _, _, m in regs]
    out, cob = [], None
    for m in ms:  # máscara vazia ou já coberta pela metade não entra
        if not m.any() or (cob is not None and (m & cob).sum() >= 0.5 * m.sum()):
            continue
        out.append(m)
        cob = m.copy() if cob is None else (cob | m)
    return out


def pair_masks(gray: np.ndarray, B: np.ndarray, cfg: PairConfig = PairConfig(), texto: np.ndarray | None = None) -> tuple[list, dict]:
    """Figura em tons de cinza + caixas do SAM -> (máscaras dos pares fortes, informações). [] = os pares não se aplicam.
    texto: máscara de texto (cmfd.textfilter, como na procura dentro da figura): o par cujos pontos concordantes caem
    em cima de letras e números sai (rótulos iguais repetidos na figura não são cópia)."""
    cx, ps, ach = candidatos(gray, B, cfg)
    info = dict(pares_paineis=len(cx), pares_melhor=0, pares_ficam=0)
    if ach and texto is not None:
        from .textfilter import on_text
        fica = []
        for x in ach:
            Sa = _para_figura(ps[x[1]], x[5], False)
            Tb = _para_figura(ps[x[2]], cv2.transform(np.asarray(x[5], np.float64)[None], x[4])[0], x[3])
            if not on_text(texto, Sa, Tb, 1.0, shape=gray.shape):
                fica.append(x)
        info["pares_texto"] = len(ach) - len(fica)
        ach = fica
    if not ach:
        return [], info
    if cfg.sem_auto or cfg.min_cz > 0:
        gf = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
        min_d = max(cfg.min_desl_px, cfg.min_desl_frac * np.hypot(*gray.shape))
        fica = []
        for x in ach:
            desl, cz = confere(gray, gf, ps[x[1]], ps[x[2]], x[3], x[4], x[5] if cfg.cz_casco else None)
            if (cfg.sem_auto and desl < min_d) or cz < cfg.min_cz:
                continue
            fica.append(x)
        ach = fica
        if not ach:
            return [], info
    regs = [(x[0], x[1], x[2], _regiao(gray.shape, ps[x[1]], ps[x[2]], x[3], x[4])) for x in ach]
    if cfg.min_area > 0:  # cópia pequena demais perto das fotos não vale (medido no treino)
        regs = [r for r in regs if r[3].sum() >= cfg.min_area * 2 * min(_area(cx[r[1]]), _area(cx[r[2]]))]
        if not regs:
            return [], info
    regs = [r for r in regs if r[0] >= cfg.rel * regs[0][0]]
    info.update(pares_melhor=int(regs[0][0]), pares_ficam=len(regs))
    return monta([r[1:] for r in regs], cfg.agrupa), info
