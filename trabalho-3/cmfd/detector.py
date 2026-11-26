"""Detector clássico de copy-move: SIFT auto-casado + espelho + RANSAC + máscara densa."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from .panels import panel_map


@dataclass
class Config:
    # pontos-chave
    max_side: int = 2048            # imagens maiores são reduzidas para este lado
    min_side_upscale: int = 512     # imagens menores são ampliadas (ideia do 2º lugar)
    contrast_threshold: float = 0.005  # baixo, como o 2º lugar: blots e fundos lisos geram poucos pontos
    edge_threshold: float = 10.0
    max_keypoints: int = 8000
    rootsift: bool = True
    sift_impl: str = "opencv"       # "opencv" (CPU) ou "kornia" (cmfd/ksift.py, GPU; pontos diferentes dos do OpenCV)
    mirror: bool = True             # ideia 2
    # casamento
    knn: int = 8
    g2nn_ratio: float = 0.6
    matcher: str = "bf"             # "bf" (cv2.BFMatcher, o da avaliação) ou "exact" (cmfd/knn.py, mesma resposta, bem mais rápido)
    min_dist_frac: float = 0.02     # pares mais próximos que isto (× diagonal) são ignorados
    min_dist_px: float = 12.0
    panel_mode: str = "all"         # ideia 1: "all" (dentro e entre), "inter", "intra"
    # RANSAC multi-modelo
    ransac_thresh: float = 3.0      # px (na escala de trabalho)
    ransac_iters: int = 4000
    max_models: int = 6
    floor_inliers: int = 5          # modelos com menos que isto nem são guardados
    min_scale: float = 0.25
    max_scale: float = 4.0
    max_aniso: float = 3.0
    # máscara (ideia 3)
    mask_mode: str = "dense"        # "bbox", "hull" ou "dense"
    zncc_win: int = 7
    zncc_thr: float = 0.8
    flat_std: float = 2.0
    hull_margin_frac: float = 1.0   # crescimento máximo além do casco dos inliers (× lado do casco)
    min_comp_frac: float = 0.0002   # componentes consistentes menores que isto são ruído
    dense_fill: str = "hull"        # "hull" (casco convexo) ou "holes" (fecha buracos)
    zncc_win2: int = 0              # 2ª janela, maior, para regiões lisas/borradas (0 = desligada)
    grow_frac: float = 0.0          # crescimento geodésico para pixels onde I(x) ≈ I(Ax) (× lado do casco)
    grow_tol: float = 6.0           # tolerância em níveis de cinza para "mesmo tom" (após suavizar)
    # candidatos extras de um ViT pré-treinado e congelado (sem treino; ver cmfd/deep_match.py)
    points: bool = True             # False = sem SIFT: só os candidatos do ViT (pipeline alternativo)
    deep: bool = False
    deep_model: str = "facebook/dinov2-small"
    deep_target: int = 672          # lado maior da imagem enviada ao ViT (patch de 14 px)
    deep_ratio: float = 0.9
    deep_min_sim: float = 0.6
    deep_ransac_frac: float = 0.6   # limiar do RANSAC em frações do tamanho do patch
    ecc_refine: bool = True         # refina o movimento grosseiro com alinhamento ECC
    deep_cascade: bool = True       # só usa o ViT se o SIFT não achou movimento firme
    cascade_min_inliers: int = 8
    deep_max_models: int = 3
    # candidatos extras do campo denso de correspondências (Cozzolino et al. 2015; ver cmfd/dense_field.py)
    field: bool = False
    field_cascade: bool = True      # só roda se o SIFT não achou movimento firme
    field_max_side: int = 768       # o campo roda na resolução nativa, reduzida até este lado
    field_radius: int = 8           # raio do disco dos momentos de Zernike
    field_iter: int = 6
    field_dlf_thr: float = 1.0      # erro médio (px²) máximo do ajuste linear local do campo
    field_max_models: int = 4
    # sementes verificadas: pares do SIFT fora de qualquer modelo, aceitos um a um pela correlação do ruído
    vseed: bool = False
    vseed_radius: int = 8           # raio do patch (px nativos) da verificação
    vseed_thr: float = 0.8          # NCC mínima do resíduo para o par valer
    vseed_max_models: int = 3
    # CLAHE (equalização local de contraste) só na imagem que vai ao SIFT, como o 3º lugar; 0 = desligado
    clahe: float = 0.0
    # candidatos por voto de Hough, sem teste de razão (cmfd/hough.py); desligado no fluxo final
    hough: bool = False
    hough_k: int = 2
    hough_min_pairs: int = 3
    # ligações sem razão confirmadas nos pixels (cmfd/hough.py: verified_models); 0 = desligado, senão k vizinhos
    vlinks_k: int = 0
    vlinks_thr: float = 0.8
    vlinks_max: int = 8             # nº máximo de grupos (candidatos) por escala
    vlinks_radius: int = 8          # raio (px originais) do recorte em que cada ligação é confirmada
    # ligações entre pedaços fixos da figura (cmfd/tiles.py)
    tiles: bool = False
    tile_ratio: float = 0.75        # 1º vizinho / 2º vizinho, dentro do outro pedaço (o mesmo da comparação entre fotos)
    tile_min_pairs: int = 4         # ligações distintas concordando com o movimento
    tile_max: int = 8               # nº máximo de candidatos por escala
    vlinks_sigma: float = 1.0       # borrão que define o "ruído fino" da confirmação (1 = ruído; 2 a 4 = textura fina, resiste a reamostragem)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------- utilidades
TEMPOS: dict = {}  # cronômetro por etapa (segundos acumulados por imagem); pipeline.predict zera e copia


def _cron(nome: str, t0: float) -> None:
    TEMPOS[nome] = TEMPOS.get(nome, 0.0) + time.time() - t0


def _prepare(gray: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
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


def _outro_detector(gray: np.ndarray, cfg: Config):
    """KAZE, AKAZE ou ORB no lugar do SIFT (fora do fluxo final)."""
    if cfg.sift_impl == "kaze":
        det = cv2.KAZE_create(threshold=1e-4)
    elif cfg.sift_impl == "akaze":
        det = cv2.AKAZE_create(threshold=1e-4)
    else:
        det = cv2.ORB_create(nfeatures=cfg.max_keypoints, fastThreshold=5, scaleFactor=1.2, nlevels=8)
    kps, des = det.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 64), np.float32)
    if len(kps) > cfg.max_keypoints:  # ficam os de maior resposta
        o = np.argsort([-k.response for k in kps])[:cfg.max_keypoints]
        kps, des = [kps[i] for i in o], des[o]
    pts = np.float32([k.pt for k in kps])
    if des.dtype == np.uint8:
        des = np.unpackbits(des, axis=1).astype(np.float32) * 2.0 - 1.0
    des = des.astype(np.float32)
    des /= np.linalg.norm(des, axis=1, keepdims=True) + 1e-7
    return pts, des


def _sift(gray: np.ndarray, cfg: Config):
    if cfg.sift_impl == "kornia":
        from .ksift import sift_kornia
        return sift_kornia(gray, cfg.max_keypoints)
    if cfg.sift_impl in ("kaze", "akaze", "orb"):
        return _outro_detector(gray, cfg)
    sift = cv2.SIFT_create(nfeatures=cfg.max_keypoints, contrastThreshold=cfg.contrast_threshold,
                           edgeThreshold=cfg.edge_threshold)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.float32([k.pt for k in kps])
    des = des.astype(np.float32)
    if cfg.rootsift:
        des /= des.sum(1, keepdims=True) + 1e-7
        des = np.sqrt(des)
    return pts, des


# ------------------------------------------------------- casamento g2NN
def self_matches(gray: np.ndarray, cfg: Config, panels: np.ndarray | None = None) -> dict:
    """Pares (p, q) de pontos parecidos dentro da mesma imagem (com espelho)."""
    h, w = gray.shape
    if cfg.matcher.startswith("lg_"):  # teste: LightGlue com SIFT ou ALIKED (cmfd/lg.py); painéis como abaixo
        from .lg import self_matches_lg
        out = self_matches_lg(gray, cfg, cfg.matcher[3:])
        if len(out["P"]) and panels is not None and cfg.panel_mode != "all":
            P, Q, M = out["P"], out["Q"], out["mirror"]
            ip = panels[np.clip(P[:, 1].astype(int), 0, h - 1), np.clip(P[:, 0].astype(int), 0, w - 1)]
            iq = panels[np.clip(Q[:, 1].astype(int), 0, h - 1), np.clip(Q[:, 0].astype(int), 0, w - 1)]
            keep = (ip != iq) if cfg.panel_mode == "inter" else (ip == iq)
            out.update(P=P[keep], Q=Q[keep], mirror=M[keep])
        return out
    t0 = time.time()
    p0, d0 = _sift(gray, cfg)
    _cron("sift", t0)
    n0 = len(p0)
    pts, des, is_mirror = p0, d0, np.zeros(n0, bool)
    if cfg.mirror and n0:
        t0 = time.time()
        pm, dm = _sift(cv2.flip(gray, 1), cfg)
        _cron("sift_espelho", t0)
        if len(pm):
            pm = pm.copy()
            pm[:, 0] = (w - 1) - pm[:, 0]       # volta para as coordenadas originais
            pts = np.vstack([p0, pm])
            des = np.vstack([d0, dm])
            is_mirror = np.r_[np.zeros(n0, bool), np.ones(len(pm), bool)]
    out = dict(n_kp=int(n0), n_kp_mirror=int(len(pts) - n0), P=np.zeros((0, 2), np.float32),
               Q=np.zeros((0, 2), np.float32), mirror=np.zeros(0, bool))
    if n0 < 3 or len(pts) < 4:
        return out
    k = min(cfg.knn + 1, len(pts))
    if cfg.matcher == "exact":  # só as consultas da imagem original; linhas (índice, distância)
        from .knn import knn as _knn
        t0 = time.time()
        I, Dd = _knn(d0, des, k)
        _cron("vizinhos", t0)
        rows = (list(zip(I[i].tolist(), Dd[i].tolist())) for i in range(n0))
        if cfg.vlinks_k:  # as ligações confirmadas nos pixels (cmfd/hough.py) reaproveitam os pontos e os vizinhos
            out["_kn"] = dict(p0=p0, pts=pts, I=I, n0=n0)
        if cfg.tiles:  # as ligações entre pedaços (cmfd/tiles.py) reaproveitam os pontos e os descritores
            out["_desc"] = dict(p0=p0, d0=d0, pm=pts[n0:], dm=des[n0:])
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        rows = ([(m.trainIdx, m.distance) for m in r] for r in bf.knnMatch(d0, des, k=k))
    min_d = max(cfg.min_dist_px, cfg.min_dist_frac * np.hypot(h, w))
    P, Q, M = [], [], []
    seen = set()
    t0 = time.time()
    for i, row in enumerate(rows):
        row = [m for m in row if m[0] != i]
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
    _cron("razao", t0)
    if not P:
        return out
    P, Q, M = np.float32(P), np.float32(Q), np.array(M)
    if panels is not None and cfg.panel_mode != "all":
        ip = panels[np.clip(P[:, 1].astype(int), 0, h - 1), np.clip(P[:, 0].astype(int), 0, w - 1)]
        iq = panels[np.clip(Q[:, 1].astype(int), 0, h - 1), np.clip(Q[:, 0].astype(int), 0, w - 1)]
        keep = (ip != iq) if cfg.panel_mode == "inter" else (ip == iq)
        P, Q, M = P[keep], Q[keep], M[keep]
    out.update(P=P, Q=Q, mirror=M)
    return out


def verify_pairs(gray_full: np.ndarray, P: np.ndarray, Q: np.ndarray, M: np.ndarray, radius: int = 8,
                 search: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """NCC do resíduo passa-alta entre patches em P e em Q (coordenadas nativas), por par."""
    if len(P) == 0:
        return np.zeros(0, np.float32)
    steps = np.arange(-search, search + 1e-6, 0.5)
    if not np.allclose(steps * 2, np.round(steps * 2)):  # a grade de meio pixel só vale para busca em múltiplos de 0,5
        return verify_pairs_ref(gray_full, P, Q, M, radius, search, sigma)
    G = gray_full.astype(np.float32)
    r = G - cv2.GaussianBlur(G, (0, 0), sigma)
    n1 = 2 * radius + 1
    d = np.arange(-radius, radius + 1, dtype=np.float32)
    du, dv = np.meshgrid(d, d)
    sx = (P[:, :1] + du.ravel()[None]).astype(np.float32)
    sy = (P[:, 1:] + dv.ravel()[None]).astype(np.float32)
    a = cv2.remap(r, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float64)
    a -= a.mean(1, keepdims=True)
    saa = (a * a).sum(1)
    a3 = a.reshape(len(P), n1, n1)
    g = np.arange(-(radius + search), radius + search + 1e-6, 0.5).astype(np.float32)  # grade de meio em meio pixel
    gx, gy = np.meshgrid(g, g)
    sgn = np.where(M, -1.0, 1.0)[:, None].astype(np.float32)
    bx = (Q[:, :1] + sgn * gx.ravel()[None]).astype(np.float32)  # no espelho a grade corre ao contrário em x
    by = (Q[:, 1:] + gy.ravel()[None]).astype(np.float32)
    big = cv2.remap(r, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float64).reshape(len(P), len(g), len(g))
    best = np.full(len(P), -1.0)
    mir = np.asarray(M, bool)
    for sub, sinal in ((np.flatnonzero(~mir), 1.0), (np.flatnonzero(mir), -1.0)):
        if not len(sub):
            continue
        bg, aa, sa = big[sub], a3[sub], saa[sub]
        melhor = np.full(len(sub), -1.0)
        for oy in steps:
            y0 = int(round((search + oy) * 2))
            for ox in steps:
                # posição x = Qx + sgn*du + ox = Qx + sgn*(du + sgn*ox): no espelho o deslocamento entra com o sinal trocado
                x0 = int(round((search + sinal * ox) * 2))
                b = bg[:, y0:y0 + 2 * n1:2, x0:x0 + 2 * n1:2]  # fatia, sem cópia
                sb = b.sum((1, 2))
                sbb = np.einsum("nij,nij->n", b, b) - sb * sb / (n1 * n1)
                sab = np.einsum("nij,nij->n", aa, b)  # a tem média zero: soma de a*(b - média de b) = soma de a*b
                melhor = np.maximum(melhor, sab / np.maximum(np.sqrt(sa * np.maximum(sbb, 0.0)), 1e-6))
        best[sub] = melhor
    best[np.sqrt(saa) < 1e-3 * (n1 * n1) ** 0.5] = -1.0  # patch sem resíduo (liso/saturado): não decide
    return best.astype(np.float32)


def verify_pairs_ref(gray_full: np.ndarray, P: np.ndarray, Q: np.ndarray, M: np.ndarray, radius: int = 8,
                 search: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Versão original (uma amostragem por deslocamento), guardada para conferir a rápida."""
    if len(P) == 0:
        return np.zeros(0, np.float32)
    G = gray_full.astype(np.float32)
    r = G - cv2.GaussianBlur(G, (0, 0), sigma)
    d = np.arange(-radius, radius + 1, dtype=np.float32)
    du, dv = np.meshgrid(d, d)
    du, dv = du.ravel()[None], dv.ravel()[None]
    sx = (P[:, :1] + du).astype(np.float32)
    sy = (P[:, 1:] + dv).astype(np.float32)
    a = cv2.remap(r, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float64)
    a -= a.mean(1, keepdims=True)
    na = np.sqrt((a * a).sum(1))
    sgn = np.where(M, -1.0, 1.0)[:, None].astype(np.float32)
    best = np.full(len(P), -1.0)
    steps = np.arange(-search, search + 1e-6, 0.5)
    for oy in steps:
        for ox in steps:
            tx = (Q[:, :1] + sgn * du + ox).astype(np.float32)
            ty = (Q[:, 1:] + dv + oy).astype(np.float32)
            b = cv2.remap(r, tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float64)
            b -= b.mean(1, keepdims=True)
            nb = np.sqrt((b * b).sum(1))
            c = (a * b).sum(1) / np.maximum(na * nb, 1e-6)
            best = np.maximum(best, c)
    best[na < 1e-3 * a.shape[1] ** 0.5] = -1.0  # patch sem resíduo (liso/saturado): não decide
    return best.astype(np.float32)


def seed_models(gray_full: np.ndarray, s: float, P: np.ndarray, Q: np.ndarray, M: np.ndarray, models: list,
                cfg: "Config") -> list[dict]:
    """Movimentos de translação (ou espelho) a partir de pares isolados com ruído correlacionado."""
    if len(P) == 0:
        return []
    explained = np.zeros(len(P), bool)
    for md in models:  # pares já explicados por um modelo do RANSAC (nos dois sentidos)
        A = np.asarray(md["A"], np.float64)
        e1 = np.hypot(*((A[:, :2] @ P.T + A[:, 2:]).T - Q).T)
        e2 = np.hypot(*((A[:, :2] @ Q.T + A[:, 2:]).T - P).T)
        explained |= np.minimum(e1, e2) < 3 * cfg.ransac_thresh
    idx = np.flatnonzero(~explained)
    if len(idx) == 0:
        return []
    ncc = verify_pairs(gray_full, P[idx] / s, Q[idx] / s, M[idx], cfg.vseed_radius)
    ok = ncc >= cfg.vseed_thr
    idx, ncc = idx[ok], ncc[ok]
    if len(idx) == 0:
        return []
    order = np.argsort(-ncc)
    idx, ncc = idx[order], ncc[order]
    t = Q[idx] - np.where(M[idx, None], np.c_[-P[idx, :1], P[idx, 1:]], P[idx])  # vetor do movimento
    tol = max(3.0 * cfg.ransac_thresh, 0.01 * np.hypot(*gray_full.shape[::-1]) * s)
    used = np.zeros(len(idx), bool)
    out = []
    rw = cfg.vseed_radius * s  # raio do patch na escala de trabalho
    corners = np.array([[-rw, -rw], [rw, -rw], [rw, rw], [-rw, rw]], np.float32)
    for k in range(len(idx)):
        if used[k] or len(out) >= cfg.vseed_max_models:
            continue
        same = (~used) & (M[idx] == M[idx[k]]) & (np.hypot(*(t - t[k]).T) < tol)
        used |= same
        m = bool(M[idx[k]])
        tv = np.median(t[same], 0)
        A = np.array([[-1.0 if m else 1.0, 0.0, tv[0]], [0.0, 1.0, tv[1]]])
        Sp = P[idx[same]]
        S = (Sp[:, None, :] + corners[None]).reshape(-1, 2).astype(np.float32)
        T = (A[:, :2] @ S.T + A[:, 2:]).T.astype(np.float32)
        n = int(same.sum())
        out.append(dict(A=A, S=S, T=T, inliers=n, n_alive=int(len(idx)), inlier_ratio=n / max(len(idx), 1),
                        valid_geom=True, shift=float(np.hypot(*(T - S).T).mean()),
                        spread=float(np.sqrt(np.prod(S.std(0) + 1e-6))), mirror_frac=float(m), scale=1.0,
                        aniso=1.0, reflect=m, vseed=1.0, vseed_ncc=float(ncc[same].mean())))
    return out


# -------------------------------------------------- RANSAC multi-modelo
def _valid_affine(A: np.ndarray, cfg: Config) -> tuple[bool, dict]:
    L = A[:, :2]
    sv = np.linalg.svd(L, compute_uv=False)
    det = float(np.linalg.det(L))
    ok = (cfg.min_scale <= sv.min() and sv.max() <= cfg.max_scale and sv.max() / max(sv.min(), 1e-9) <= cfg.max_aniso)
    return ok, dict(scale=float(np.sqrt(abs(det))), aniso=float(sv.max() / max(sv.min(), 1e-9)), reflect=det < 0)


def snap_affine(A: np.ndarray, S: np.ndarray, T: np.ndarray, thresh: float, keep: float = 0.9) -> tuple[np.ndarray, int]:
    """Navalha de Occam na geometria: troca o afim (6 graus de liberdade) pelo modelo mais simples que ainda explica `keep` dos inliers dentro de `thresh` px."""
    if len(S) < 3:
        return A, 2
    S, T = S.astype(np.float64), T.astype(np.float64)
    L0 = A[:, :2]
    nA = int((np.hypot(*(S @ L0.T + A[:, 2] - T).T) <= thresh).sum())
    need = keep * max(nA, 1)
    det = float(np.linalg.det(L0))
    if det > 0:
        t = np.median(T - S, axis=0)
        if (np.hypot(*(S + t - T).T) <= thresh).sum() >= need:
            return np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]]]), 0
    F = np.diag([-1.0, 1.0]) if det < 0 else np.eye(2)
    use = np.ones(len(S), bool)
    for _ in range(2):  # mínimos quadrados e um refinamento sem os pares fora do limiar
        Sf = S[use] @ F.T
        ms, mt = Sf.mean(0), T[use].mean(0)
        xc = (Sf - ms) @ np.array([1.0, 1j])
        yc = (T[use] - mt) @ np.array([1.0, 1j])
        z = (np.conj(xc) @ yc) / max(float((np.abs(xc) ** 2).sum()), 1e-9)
        Lz = np.array([[z.real, -z.imag], [z.imag, z.real]])
        Asim = np.hstack([Lz @ F, (mt - Lz @ ms)[:, None]])
        r = np.hypot(*(S @ Asim[:, :2].T + Asim[:, 2] - T).T)
        if (r <= thresh).sum() < 3:
            break
        use = r <= thresh
    if (r <= thresh).sum() >= need:
        return Asim, 1
    return A, 2


def fit_models(P: np.ndarray, Q: np.ndarray, mirror: np.ndarray, cfg: Config) -> list[dict]:
    """Extrai movimentos afins até não sobrar suporte (cada par vale nos dois sentidos)."""
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
        used = fwd | bwd
        n_in = int(fwd.sum() + bwd.sum())
        if n_in < cfg.floor_inliers:
            break
        alive[idx[used]] = False
        ok, geo = _valid_affine(A, cfg)
        # pontos de origem/destino no sentido do modelo A (src -> dst)
        S = np.vstack([P[idx][fwd], Q[idx][bwd]])
        T = np.vstack([Q[idx][fwd], P[idx][bwd]])
        shift = float(np.median(np.hypot(*(T - S).T))) if len(S) else 0.0
        spread = float(np.sqrt(np.prod(S.std(0) + 1e-6))) if len(S) > 1 else 0.0
        mir_frac = float(np.r_[mirror[idx][fwd], mirror[idx][bwd]].mean()) if n_in else 0.0
        models.append(dict(A=A, S=S, T=T, inliers=n_in, n_alive=int(len(idx)),
                           inlier_ratio=n_in / max(len(idx), 1), valid_geom=bool(ok), shift=shift,
                           spread=spread, mirror_frac=mir_frac, **geo))
    return models


# ----------------------------------------------------------- máscaras
def _hull_mask(pts: np.ndarray, shape, dilate: int) -> np.ndarray:
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
        else:  # disco grande: transformada de distância (linear no nº de pixels; mesmo resultado, euclidiano)
            dist = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            m = (dist <= dilate).astype(np.uint8)
    return m


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Preenche os buracos da máscara, como scipy.ndimage.binary_fill_holes, só no retângulo dela."""
    m = np.asarray(m).astype(bool)
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if len(rows) == 0:
        return m
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    c = np.ones((y1 - y0 + 2, x1 - x0 + 2), np.uint8)  # 1 = fundo
    c[1:-1, 1:-1] = ~m[y0:y1, x0:x1]
    cv2.floodFill(c, None, (0, 0), 2, flags=4)
    out = m.copy()
    out[y0:y1, x0:x1] |= c[1:-1, 1:-1] == 1
    return out


def _bbox_mask(pts: np.ndarray, shape, margin: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if len(pts):
        x0, y0 = np.floor(pts.min(0)).astype(int) - margin
        x1, y1 = np.ceil(pts.max(0)).astype(int) + margin
        m[max(y0, 0):max(y1 + 1, 0), max(x0, 0):max(x1 + 1, 0)] = 1
    return m


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


def _geodesic_grow(core: np.ndarray, allowed: np.ndarray, radius: int) -> np.ndarray:
    """Cresce `core` dentro de `allowed` até `radius` px (dilatações sucessivas de 2 px)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = core.copy()
    allowed = (allowed | core).astype(np.uint8)
    for _ in range(max(radius // 2, 0)):
        nxt = cv2.dilate(m, k) & allowed
        if np.array_equal(nxt, m):
            break
        m = nxt
    return m


def dense_masks(gray: np.ndarray, model: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras (origem, cópia) pela sobreposição: ZNCC local entre I(x) e I(A x)."""
    if cfg.dense_fill != "holes" or cfg.grow_frac > 0 or len(model["S"]) == 0:
        return _dense_masks_full(gray, model, cfg)
    h, w = gray.shape
    A = model["A"]
    G = gray.astype(np.float32)
    J = cv2.warpAffine(G, A, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), A, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    J = np.where(valid, J, 0).astype(np.float32)
    S = model["S"]
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
    if cfg.zncc_win2:
        cand |= _zncc(Gc, Jc, vc, cfg.zncc_win2, cfg.flat_std) > cfg.zncc_thr
    cand = cand.astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    Sc = (S - np.array([x0, y0], S.dtype)).astype(S.dtype)
    cand &= _hull_mask(Sc, cand.shape, margin)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    min_px = max(cfg.zncc_win * cfg.zncc_win, int(cfg.min_comp_frac * h * w))
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
    if not keep:
        src = _hull_mask(S, (h, w), cfg.zncc_win // 2)
    else:
        sc = np.isin(lab, keep).astype(np.uint8)
        sc = cv2.morphologyEx(sc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kc + 1, 2 * kc + 1)))
        src = np.zeros((h, w), np.uint8)
        src[y0:y1, x0:x1] = fill_holes(sc)
    dst = cv2.warpAffine(src, A, (w, h), flags=cv2.INTER_NEAREST)
    return src, dst


def _dense_masks_full(gray: np.ndarray, model: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Versão de referência de dense_masks, na imagem inteira."""
    h, w = gray.shape
    A = model["A"]
    G = gray.astype(np.float32)
    J = cv2.warpAffine(G, A, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), A, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    J = np.where(valid, J, 0).astype(np.float32)
    cand = _zncc(G, J, valid, cfg.zncc_win, cfg.flat_std) > cfg.zncc_thr
    if cfg.zncc_win2:  # janela grande: pega estrutura de baixa frequência (bandas borradas, ampliação)
        cand |= _zncc(G, J, valid, cfg.zncc_win2, cfg.flat_std) > cfg.zncc_thr
    cand = cand.astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # região de busca generosa em volta dos inliers da origem: a cópia costuma ir
    # além dos pontos (bordas e fundo liso não geram pontos-chave)
    S = model["S"]
    side = np.sqrt(max(cv2.contourArea(cv2.convexHull(S.reshape(-1, 1, 2))) if len(S) >= 3 else 0, 1.0))
    margin = int(max(4 * cfg.zncc_win, cfg.hull_margin_frac * side))
    region = _hull_mask(S, (h, w), margin)
    cand &= region
    # fora da cópia, ZNCC alto em pixel com textura é raro: sobram só manchas pequenas
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    min_px = max(cfg.zncc_win * cfg.zncc_win, int(cfg.min_comp_frac * h * w))
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
    if not keep:
        src = _hull_mask(S, (h, w), cfg.zncc_win // 2)
    else:
        src = np.isin(lab, keep).astype(np.uint8)
        if cfg.dense_fill == "hull":  # áreas lisas dentro da cópia entram pelo casco convexo
            pts = np.column_stack(np.nonzero(src)[::-1]).astype(np.float32)
            src = _hull_mask(np.vstack([pts, S]), (h, w), 0)
        else:
            k = max(3, cfg.zncc_win) | 1
            src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            src = fill_holes(src).astype(np.uint8)
    if cfg.grow_frac > 0:
        # fundo liso e bordas: ZNCC não decide (sem textura), mas se o tom em x e em A x é o
        # mesmo, o pixel é consistente com a cópia; cresce a partir do núcleo, com alcance limitado
        bg = cv2.GaussianBlur(G, (0, 0), 1.5)
        bj = cv2.GaussianBlur(J, (0, 0), 1.5)
        agree = ((np.abs(bg - bj) <= cfg.grow_tol) & valid).astype(np.uint8) & region
        src = _geodesic_grow(src, agree, int(round(cfg.grow_frac * side)))
        src = fill_holes(src).astype(np.uint8)
    dst = cv2.warpAffine(src, A, (w, h), flags=cv2.INTER_NEAREST)
    return src, dst


def residual_ncc(gray: np.ndarray, A: np.ndarray, src: np.ndarray, sigma: float = 1.0, search: float = 1.0,
                 max_pts: int = 20000, return_offset: bool = False, r: np.ndarray | None = None):
    """Correlação do RUÍDO entre origem e cópia: NCC do resíduo passa-alta em x e em A x, x na máscara de origem."""
    if src.shape != gray.shape:
        raise ValueError(f"máscara {src.shape} e imagem {gray.shape} com formas diferentes")
    ys, xs = np.nonzero(src)
    if len(ys) < 16:
        return (0.0, (0.0, 0.0)) if return_offset else 0.0
    if len(ys) > max_pts:
        sel = np.random.default_rng(0).choice(len(ys), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    if r is None:
        r = noise_residual(gray, sigma)
    a = r[ys, xs].astype(np.float64)
    a -= a.mean()
    na = np.sqrt((a * a).sum())
    if na < 1e-3:
        return (0.0, (0.0, 0.0)) if return_offset else 0.0
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
    return (best, off) if return_offset else best


def noise_residual(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    G = gray.astype(np.float32)
    return G - cv2.GaussianBlur(G, (0, 0), sigma)


def noise_refine(gray: np.ndarray, A: np.ndarray, src: np.ndarray, win: int = 9, thr: float = 0.4,
                 low: float = 0.1, min_std: float = 0.5, grow_frac: float = 0.5, sigma: float = 1.0,
                 iters: int = 3, gate: float = 0.3, r: np.ndarray | None = None, pre=None) -> np.ndarray:
    """Refina a máscara de origem pelo mapa LOCAL de correlação do ruído entre x e A x (resolução nativa)."""
    H, W = gray.shape
    src = src.astype(bool)
    if src.sum() < 16:
        return src
    if r is None:
        r = noise_residual(gray, sigma)
    rn, (dx, dy) = pre if pre is not None else residual_ncc(gray, A, src, sigma=sigma, return_offset=True, r=r)
    if rn < gate:  # sem ruído copiado claro (cópia reamostrada/comprimida, ou repetição honesta): não mexe
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
    A2[:, 2] += A2[:, :2] @ np.array([x0, y0], np.float64)  # pixel (u, v) do recorte = (u + x0, v + y0)
    rc = np.ascontiguousarray(r[y0:y1, x0:x1])
    J = cv2.warpAffine(r, A2, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
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
    for _ in range(iters):  # cresce em passos limitados, sempre conectado ao núcleo
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


def model_masks(gray: np.ndarray, model: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    shape = gray.shape
    if cfg.mask_mode == "bbox":
        m = max(2, cfg.zncc_win // 2)
        return _bbox_mask(model["S"], shape, m), _bbox_mask(model["T"], shape, m)
    if cfg.mask_mode == "hull":
        d = max(2, cfg.zncc_win // 2)
        return _hull_mask(model["S"], shape, d), _hull_mask(model["T"], shape, d)
    return dense_masks(gray, model, cfg)


# ------------------------------------------------------------ pipeline
def detect_candidates(gray_full: np.ndarray, cfg: Config, masks: bool = True) -> dict:
    """Parte cara: pontos, casamentos, modelos e máscaras de TODOS os modelos candidatos."""
    t0 = time.time()
    H, W = gray_full.shape
    gray, s = _prepare(gray_full, cfg)
    panels = panel_map(gray) if cfg.panel_mode != "all" else None
    if cfg.points:
        gk = cv2.createCLAHE(clipLimit=cfg.clahe, tileGridSize=(8, 8)).apply(gray) if cfg.clahe > 0 else gray
        m = self_matches(gk, cfg, panels)
    else:
        m = dict(n_kp=0, n_kp_mirror=0, P=np.zeros((0, 2), np.float32), Q=np.zeros((0, 2), np.float32),
                 mirror=np.zeros(0, bool))
    t0 = time.time()
    models = fit_models(m["P"], m["Q"], m["mirror"], cfg)
    _cron("movimentos", t0)
    if cfg.hough and cfg.points:
        from .hough import hough_models
        models += hough_models(gk, cfg, cfg.hough_k, cfg.hough_min_pairs)
    for md in models:
        md["deep"] = 0.0
    n_deep = 0
    strong = any(md["inliers"] >= cfg.cascade_min_inliers and md["inlier_ratio"] >= 0.3 and md["valid_geom"]
                 and md.get("aniso", 1.0) <= 1.5 for md in models)
    if cfg.deep and not (cfg.deep_cascade and strong):  # cascata: o ViT só entra onde o SIFT não achou nada firme
        from dataclasses import replace as _rep

        from .deep_match import deep_matches
        P2, Q2, pz = deep_matches(gray, cfg.deep_model, cfg.deep_target, cfg.deep_ratio, cfg.deep_min_sim)
        n_deep = int(len(P2))
        dcfg = _rep(cfg, ransac_thresh=cfg.deep_ransac_frac * pz, floor_inliers=4, max_models=cfg.deep_max_models)
        for md in fit_models(P2, Q2, np.zeros(len(P2), bool), dcfg):
            if cfg.ecc_refine:
                refine_ecc(gray, md)
            md["deep"] = 1.0
            models.append(md)
    if cfg.vseed:
        for md in seed_models(gray_full, s, m["P"], m["Q"], m["mirror"], models, cfg):
            if cfg.ecc_refine:
                refine_ecc(gray, md)
            md["deep"] = 0.0
            models.append(md)
    if cfg.tiles and cfg.points and "_desc" in m:
        from .tiles import tile_models
        t0 = time.time()
        for md in tile_models(m["_desc"]["p0"], m["_desc"]["d0"], m["_desc"]["pm"], m["_desc"]["dm"], gk.shape, cfg):
            md["deep"] = 0.0
            models.append(md)
        _cron("pedacos", t0)
    if cfg.vlinks_k and cfg.points:
        from .hough import verified_models
        t0 = time.time()
        for md in verified_models(gray_full, gk, s, cfg, cfg.vlinks_k, cfg.vlinks_thr, radius=cfg.vlinks_radius, max_models=cfg.vlinks_max, kn=m.get("_kn"), sigma=cfg.vlinks_sigma):
            md["deep"] = 0.0
            models.append(md)
        _cron("ligacoes", t0)
    if cfg.field and not (cfg.field_cascade and strong):
        from .dense_field import field_models
        fms, _ = field_models(gray_full, cfg.field_max_side, cfg.field_radius, n_iter=cfg.field_iter,
                              dlf_thr=cfg.field_dlf_thr, max_models=cfg.field_max_models)
        for md in fms:  # coordenadas nativas -> escala de trabalho
            md["A"][:, 2] *= s
            md["S"], md["T"] = md["S"] * s, md["T"] * s
            md["shift"], md["spread"] = md["shift"] * s, md["spread"] * s
            ok, geo = _valid_affine(md["A"], cfg)
            md.update(geo, valid_geom=bool(ok), deep=0.0, field=1.0)
            if cfg.ecc_refine:
                refine_ecc(gray, md)
            models.append(md)
    cands = []
    for md in models:
        if not masks:
            stats = {k: v for k, v in md.items() if k not in ("A", "S", "T")}
            stats["shift"] = stats["shift"] / s
            stats["spread"] = stats["spread"] / s
            cands.append(dict(stats=stats, src=None, dst=None, A=np.asarray(md["A"], np.float32),
                              S=np.asarray(md["S"], np.float32), T=np.asarray(md["T"], np.float32)))
            continue
        src, dst = model_masks(gray, md, cfg)
        if s != 1.0:
            src = cv2.resize(src, (W, H), interpolation=cv2.INTER_NEAREST)
            dst = cv2.resize(dst, (W, H), interpolation=cv2.INTER_NEAREST)
        stats = {k: v for k, v in md.items() if k not in ("A", "S", "T")}
        stats["src_area"] = float(src.mean())
        stats["dst_area"] = float(dst.mean())
        inter = float((src & dst).sum()) / max(float((src | dst).sum()), 1.0)
        stats["src_dst_iou"] = inter
        An = np.asarray(md["A"], np.float64).copy()
        An[:, 2] /= s  # movimento na escala nativa
        stats["resid_ncc"] = residual_ncc(gray_full, An, src.astype(bool))
        stats["shift"] = stats["shift"] / s
        stats["spread"] = stats["spread"] / s
        cands.append(dict(stats=stats, src=src.astype(bool), dst=dst.astype(bool),
                          A=np.asarray(md["A"], np.float32), S=np.asarray(md["S"], np.float32), T=np.asarray(md["T"], np.float32)))
    return dict(cands=cands, n_kp=m["n_kp"], n_kp_mirror=m["n_kp_mirror"], n_matches=int(len(m["P"])),
                n_deep_matches=n_deep, scale=s, shape=(H, W), seconds=time.time() - t0)


def refine_ecc(gray: np.ndarray, md: dict, max_side: int = 640) -> None:
    """Refina o movimento afim (origem -> cópia) por alinhamento ECC na região da origem."""
    h, w = gray.shape
    s = min(1.0, max_side / max(h, w))
    g = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    S = md["S"] * s
    side = np.sqrt(max(cv2.contourArea(cv2.convexHull(S.reshape(-1, 1, 2))) if len(S) >= 3 else 0, 64.0))
    mask = _hull_mask(S, g.shape, int(max(8, 0.5 * side)))
    G = g.astype(np.float32)
    warp = md["A"].astype(np.float32).copy()
    warp[:, 2] *= s
    try:
        _, warp = cv2.findTransformECC(G, G, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), mask, 5)
    except cv2.error:
        return
    A = warp.astype(np.float64)
    A[:, 2] /= s
    S = md["S"]
    if np.abs(A - md["A"]).max() > 0.25 * max(h, w):  # divergiu: mantém o original
        return
    md["A"] = A
    md["T"] = (A[:, :2] @ S.T + A[:, 2:]).T.astype(np.float32)


@dataclass
class Selection:
    """Limiares baratos aplicados aos movimentos candidatos."""
    min_inliers: int = 12
    min_inlier_ratio: float = 0.0
    min_shift_frac: float = 0.03     # deslocamento mínimo (× diagonal)
    min_area_frac: float = 0.0005    # área mínima de cada máscara (× imagem)
    max_src_dst_iou: float = 0.5     # origem e cópia não podem se sobrepor demais
    max_aniso: float = 3.0           # razão entre os valores singulares da parte linear
    deep_min_inliers: int = 1000     # limiar próprio para candidatos do ViT (1000 = desligado)
    min_rel_inliers: float = 0.0     # descarta movimentos com menos que esta fração dos inliers do mais forte
    field_min_support: float = 1.0   # candidatos do campo denso: fração mínima da imagem coerente (1 = desligado)
    max_degree: int = 99             # região casada com >= este nº de outras = padrão repetido (rótulos, poços)
    min_resid_ncc: float = -1.0      # correlação mínima do ruído entre origem e cópia (-1 = desligado)
    vseed_min_inliers: int = 1000    # sementes verificadas pelo ruído: nº mínimo de pares (1000 = desligadas)
    vseed_min_margin: float = -1.0   # sementes: concordância de sinal do ruído no pico menos o maior lóbulo lateral (cmfd/peak.py); -1 = desligado
    tile_min_inliers: int = 0        # candidatos de pedaços fixos (cmfd/tiles.py): mínimo de inliers (2 por ligação); 0 = sem corte próprio
    tile_min_margin: float = -1.0    # idem: checagem de sinal no pico, como nas sementes; -1 = desligada
    pico_todos: bool = False         # calcula as medidas do pico e da região para TODO candidato (tabela do verificador), sem cortar por elas
    vseed_min_frac: float = 0.5      # sementes: fração mínima da região com ruído mensurável (só vale com vseed_min_margin ligado)
    resid_bypass_inliers: int = 100000  # geometria forte dispensa o teste do ruído (cópia reamostrada/comprimida)
    max_bg_frac: float = 1.0         # candidato com pontos sobre o fundo da figura (texto, linhas, gráficos); 1 = desligado
    dedup_iou: float = 1.0           # candidato com IoU > isto com um mais forte é descartado (não somado); 1 = desligado
    require_valid_geom: bool = True
    instances: str = "pair"          # "pair" (origem e cópia separadas), "union" (por modelo), "single"
    max_masks: int = 4
    merge_iou: float = 0.5
    merge_frac: float = 0.0          # merge_iou = 0: junta só se a interseção > isto × menor área (0 = qualquer toque)
    extra: dict = field(default_factory=dict)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


def _overlaps(res: dict) -> np.ndarray:
    """Matriz (2n x 2n) de sobreposição entre as máscaras origem/cópia de todos os candidatos (memorizada)."""
    if "_ov" not in res:
        ms = []
        for c in res["cands"]:
            ms += [c["src"][::4, ::4], c["dst"][::4, ::4]]
        n = len(ms)
        area = np.array([m.sum() for m in ms], np.float64)
        ov = np.zeros((n, n), bool)
        for a in range(n):
            for b in range(a + 1, n):
                inter = np.logical_and(ms[a], ms[b]).sum()
                ov[a, b] = ov[b, a] = inter > 0.3 * max(min(area[a], area[b]), 1.0)
        res["_ov"] = ov
    return res["_ov"]


def _drop_repeated(res: dict, keep: list[int], max_degree: int) -> list[int]:
    """Tira candidatos cujas regiões casam com >= max_degree outras regiões (padrão repetido honesto)."""
    ov = _overlaps(res)
    nodes = [2 * i for i in keep] + [2 * i + 1 for i in keep]
    parent = {u: u for u in nodes}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for a in nodes:
        for b in nodes:
            if a < b and ov[a, b]:
                parent[find(a)] = find(b)
    nbr: dict[int, set] = {}
    for i in keep:
        ra, rb = find(2 * i), find(2 * i + 1)
        if ra != rb:
            nbr.setdefault(ra, set()).add(rb)
            nbr.setdefault(rb, set()).add(ra)
    bad = {r for r, s in nbr.items() if len(s) >= max_degree}
    if not bad:
        return keep
    # espalha a marca pelo componente do grafo (todas as cópias do padrão repetido)
    stack = list(bad)
    while stack:
        r = stack.pop()
        for q in nbr.get(r, ()):
            if q not in bad:
                bad.add(q)
                stack.append(q)
    return [i for i in keep if find(2 * i) not in bad and find(2 * i + 1) not in bad]


def passing(res: dict, sel: Selection) -> tuple[int, ...]:
    """Índices dos candidatos que passam nos filtros (ordenados por nº de inliers)."""
    H, W = res["shape"]
    diag = np.hypot(H, W)
    keep = []
    for i, c in enumerate(res["cands"]):
        st = c["stats"]
        if st.get("field", 0.0):
            if st.get("support_frac", 0.0) < sel.field_min_support:
                continue
        elif st.get("vseed", 0.0):
            if st["inliers"] < sel.vseed_min_inliers:
                continue
            if sel.vseed_min_margin >= 0 and (st.get("s_margem", 1.0) < sel.vseed_min_margin
                                              or st.get("s_frac", 1.0) < sel.vseed_min_frac):
                continue
        else:
            need = sel.deep_min_inliers if st.get("deep", 0.0) else sel.min_inliers
            if st["inliers"] < need:
                continue
            if st.get("tile", 0.0):
                if st["inliers"] < sel.tile_min_inliers:
                    continue
                if sel.tile_min_margin >= 0 and (st.get("s_margem", 1.0) < sel.tile_min_margin or st.get("s_frac", 1.0) < sel.vseed_min_frac):
                    continue
        if st["inlier_ratio"] < sel.min_inlier_ratio:
            continue
        if sel.require_valid_geom and not st["valid_geom"]:
            continue
        if st.get("aniso", 1.0) > sel.max_aniso:
            continue
        if st["shift"] < sel.min_shift_frac * diag:
            continue
        if st["src_area"] < sel.min_area_frac or st["dst_area"] < sel.min_area_frac:
            continue
        if st["src_dst_iou"] > sel.max_src_dst_iou:
            continue
        if st.get("bg_frac", 0.0) > sel.max_bg_frac:
            continue
        if st.get("resid_ncc", 1.0) < sel.min_resid_ncc and not (
                st["inliers"] >= sel.resid_bypass_inliers and not st.get("field", 0.0)):
            continue
        keep.append(i)
    # pontos (SIFT/ViT) primeiro, por nº de inliers; depois o campo denso, por suporte
    def fam(i):  # família do candidato: 0 = pontos (SIFT/ViT), 1 = sementes verificadas, 2 = campo denso
        st = res["cands"][i]["stats"]
        return 2 if st.get("field", 0.0) else (1 if st.get("vseed", 0.0) else 0)
    keep.sort(key=lambda i: (fam(i), -res["cands"][i]["stats"]["inliers"]))
    if keep and sel.min_rel_inliers > 0:  # movimentos fracos ao lado de um forte (da mesma família) são pedaços/ruído
        top = {}
        for i in keep:
            top.setdefault(fam(i), res["cands"][i]["stats"]["inliers"])
        keep = [i for i in keep if res["cands"][i]["stats"]["inliers"] >= sel.min_rel_inliers * top[fam(i)]]
    if keep and sel.max_degree < 99:
        keep = _drop_repeated(res, keep, sel.max_degree)
    return tuple(keep)


def select(res: dict, sel: Selection) -> list[np.ndarray]:
    """Parte barata: filtra os candidatos e monta a lista final de máscaras ([] = authentic)."""
    return compose(res, passing(res, sel), sel)


def compose(res: dict, idxs: tuple[int, ...], sel: Selection) -> list[np.ndarray]:
    """Monta as máscaras finais a partir dos candidatos escolhidos."""
    chosen = [res["cands"][i] for i in idxs]
    parts = []
    for c in chosen:
        new = [c["src"], c["dst"]] if sel.instances == "pair" else [c["src"] | c["dst"]]
        if sel.dedup_iou < 1.0 and parts:
            # o mesmo evento achado de novo (outra escala/modelo vizinho): fica a máscara do candidato mais forte
            new = [p for p in new if max(_iou(p, q) for q in parts) <= sel.dedup_iou]
        parts += new
    masks: list[np.ndarray] = []
    if sel.merge_iou <= 0 and parts:
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
                    # o gabarito nunca tem máscaras sobrepostas: um toque na borda pode ser dois eventos vizinhos
                    if inter > 0 and inter > sel.merge_frac * min(area[a], area[b]):
                        parent[find(a)] = find(b)
        groups: dict[int, np.ndarray] = {}
        for k in range(n):  # mantém a ordem (o grupo do candidato mais forte primeiro)
            r = find(k)
            groups[r] = groups[r] | parts[k] if r in groups else parts[k].copy()
        masks = list(groups.values())
    else:
        for p in parts:
            dup = next((k for k, m in enumerate(masks) if _iou(m, p) > sel.merge_iou), None)
            if dup is None:
                masks.append(p.copy())
            else:
                masks[dup] |= p
    if sel.instances == "single" and masks:
        u = masks[0].copy()
        for m in masks[1:]:
            u |= m
        masks = [u]
    return [m for m in masks if m.any()][:sel.max_masks]
