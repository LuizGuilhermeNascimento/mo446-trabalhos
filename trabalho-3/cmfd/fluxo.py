"""Fluxo final: do arquivo da figura à resposta.

Faz, sem gabarito, as mesmas contas dos scripts de treino: scripts/verif_cache.py (candidatos e medidas),
scripts/verif_treina.py (verificador) e scripts/mascara_seletor.py (seletor de máscara). Configuração: comparação
entre fotos; se ela não responde, SIFT em 3 ampliações (Kornia, na GPU), verificador com corte 0,9 sem regras fixas,
seletor com limiar 0,02.
"""
from __future__ import annotations

import pickle
from dataclasses import replace

import cv2
import numpy as np

from . import pipeline
from .detector import compose, detect_candidates
from .pairs import PairConfig, caixas, pair_masks
from .panelfilter import load_panels
from .textfilter import load_sam_text

MEDIDAS = ["inliers", "inlier_ratio", "n_alive", "shift_rel", "spread_rel", "scale", "aniso", "reflect", "mirror_frac", "valid_geom",
           "src_area", "dst_area", "src_dst_iou", "resid_ncc", "s_margem", "s_frac", "s_pico", "s_lado_med", "cz", "tx_std", "tx_media",
           "tx_sat", "tx_grad", "area_px", "caixa_asp", "cheio", "borda", "vseed", "tile", "vseed_ncc", "origem",
           "mp", "n_cands", "pos_inliers", "rel_inliers", "tem_fotos", "tem_texto"]
COMUNS = ["q_area", "q_lado", "q_cheio", "q_frac", "n_pecas", "n_inst", "mp", "n_obj", "par_q", "regras"]
DO_ALT = ["n_toc", "rel", "iou", "cob_q", "cob_alt", "cheio", "nota", "frase", "par"]
CORTE, LIMIAR, MAX_FRAC, TOCA = 0.9, 0.02, 0.25, 0.5
PARES = PairConfig(lado_min=0, sem_auto=True)
ABERTO = replace(pipeline.SELECTION, min_resid_ncc=-1.0, max_aniso=2.0, max_src_dst_iou=0.3, min_rel_inliers=0.0,
                 vseed_min_inliers=2, vseed_min_margin=-1.0, tile_min_inliers=0, tile_min_margin=-1.0, pico_todos=True, dedup_iou=1.0)


def _prepara(sift: str) -> None:
    """Os mesmos ajustes globais de verif_cache.py: work, feitos antes da comparação entre fotos (ela lê pipeline.VARIANTS)."""
    for v in pipeline.SCALES:
        pipeline.VARIANTS[v] = replace(pipeline.VARIANTS[v], clahe=2.0, mirror=True, vlinks_k=2, vlinks_thr=0.8, vlinks_max=8, vlinks_radius=4,
                                       tiles=True, tile_ratio=0.75, tile_min_pairs=4, tile_max=16, sift_impl=sift)


def candidatos(full: np.ndarray, sam_npz: str, sift: str = "kornia") -> dict:
    """Comparação entre fotos e, se ela não responde, os candidatos das 3 ampliações com as medidas (= verif_cache.py: work)."""
    _prepara(sift)
    H, W = full.shape
    P, T = load_panels(sam_npz), load_sam_text(sam_npz)
    B = np.load(sam_npz)["B"]
    out = dict(shape=(H, W), pares=[], cands=[], n_fotos=0 if P is None else 1, tem_texto=T is not None,
               n_caixas=len(caixas(B, H, W, PARES)))
    ms = pair_masks(full, B, PARES)[0]
    if ms:
        out["pares"] = [np.asarray(m).astype(bool) for m in ms]
        return out
    rfull = None
    for vi, v in enumerate(pipeline.SCALES):
        res = detect_candidates(full, replace(pipeline.VARIANTS[v], matcher=pipeline.MATCHER, sift_impl=sift), masks=False)
        part, rfull = pipeline.remask(full, res, v, rfull, ABERTO, T, P)
        for c in part:
            if not (c["src"] | c["dst"]).any():
                continue
            out["cands"].append(dict(stats=dict(c["stats"], origem=float(vi)), src=c["src"], dst=c["dst"]))
    return out


def medidas(d: dict) -> np.ndarray:
    """As 37 medidas de cada candidato, na ordem de MEDIDAS (= verif_treina.py)."""
    H, W = d["shape"]
    ins = sorted((c["stats"]["inliers"] for c in d["cands"]), reverse=True)
    X = np.zeros((len(d["cands"]), len(MEDIDAS)), np.float32)
    for i, c in enumerate(d["cands"]):
        st = c["stats"]
        r = {m: float(st.get(m, np.nan)) for m in MEDIDAS}
        r.update(shift_rel=st["shift"] / np.hypot(H, W), spread_rel=st.get("spread", 0.0) / np.hypot(H, W), mp=H * W / 1e6, n_cands=len(d["cands"]),
                 pos_inliers=ins.index(st["inliers"]), rel_inliers=st["inliers"] / max(ins[0], 1), tem_fotos=d["n_fotos"], tem_texto=float(d["tem_texto"]))
        X[i] = [r[m] for m in MEDIDAS]
    return X


def treina_verificador(T):
    """T: tabela candidatos.csv do treino (scripts/verif_treina.py). Mesmo modelo e mesmos parâmetros do script de treino."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    return m.fit(T[MEDIDAS].to_numpy(np.float32), T.y.to_numpy(), sample_weight=np.ones(len(T)))


def _feats(P, nm):
    return P[COMUNS + [f"{nm}_{c}" for c in DO_ALT]].to_numpy(np.float32)


def treina_seletor(P):
    """P: tabela pedacos.csv do treino (scripts/mascara_seletor.py). Devolve os dois regressores (U, B)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    f = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    return f().fit(_feats(P, "u"), (P.g_u - P.g_q).to_numpy()), f().fit(_feats(P, "b"), (P.g_b - P.g_q).to_numpy())


def aceitos(d: dict, probs: np.ndarray) -> list:
    """Máscaras depois do verificador (corte 0,9, sem regras fixas) e da junção."""
    if not len(d["cands"]):
        return []
    idx = [int(i) for i in np.argsort(-probs) if probs[i] >= CORTE]
    base = compose(dict(cands=d["cands"], shape=d["shape"]), tuple(idx), pipeline.SELECTION)
    return [np.asarray(m).astype(bool) for m in base if np.asarray(m).any()]


def _peq(m, hw):
    return m if m.shape == tuple(hw) else cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _gra(m, shape):
    return m if m.shape == tuple(shape) else cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _caixa(m):
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def pedacos(base: list, shape: tuple, obj_pkl: str, n_caixas: int):
    """Cada pedaço de cada máscara com as duas alternativas (U, B) e as medidas (= mascara_seletor.py: work, sem gabarito)."""
    z = pickle.load(open(obj_pkl, "rb"))
    hw = tuple(z["hw"])
    objs = []
    for o in z["objs"]:
        y0, y1, x0, x1 = o["caixa"]
        cr = np.unpackbits(np.frombuffer(o["bits"], np.uint8))[:(y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)
        ar = int(cr.sum())
        if 0 < ar <= MAX_FRAC * hw[0] * hw[1]:
            objs.append((o["frase"], o["nota"], (y0, y1, x0, x1), cr, ar))
    rows, alts = [], {}
    for i, m in enumerate(base):
        ms = _peq(m, hw)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ms.astype(np.uint8), connectivity=8)
        maior = st[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
        ks = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= max(9, 0.02 * maior)]
        da_inst = []
        for k in ks:
            q = lab == k
            aq = int(st[k, cv2.CC_STAT_AREA])
            x, y, w, h = (int(v) for v in st[k, :4])
            toc, melhor = [], None
            for f, nt, (y0, y1, x0, x1), cr, ar in objs:
                if y1 <= y or y0 >= y + h or x1 <= x or x0 >= x + w:
                    continue
                inter = int((q[y0:y1, x0:x1] & cr).sum())
                if not inter:
                    continue
                io = inter / float(aq + ar - inter)
                if melhor is None or io > melhor[0]:
                    melhor = (io, f, nt, (y0, y1, x0, x1), cr)
                if inter >= TOCA * min(ar, aq):
                    toc.append((f, nt, (y0, y1, x0, x1), cr))
            U = np.zeros(hw, bool)
            for _, _, (y0, y1, x0, x1), cr in toc:
                U[y0:y1, x0:x1] |= cr
            Bm = np.zeros(hw, bool)
            if melhor is not None:
                y0, y1, x0, x1 = melhor[3]
                Bm[y0:y1, x0:x1] = melhor[4]
            r = dict(inst=i, k=k, q_area=aq, q_lado=min(w, h), q_cheio=aq / float(w * h), q_frac=aq / float(hw[0] * hw[1]),
                     n_pecas=len(ks), n_inst=len(base), mp=shape[0] * shape[1] / 1e6, n_obj=len(objs), regras=int(n_caixas >= 2))
            for nm, alt, lst in (("u", U, toc), ("b", Bm, [melhor[1:]] if melhor else [])):
                aa = int(alt.sum())
                inter = int((alt & q).sum())
                if aa:
                    y0, y1, x0, x1 = _caixa(alt)
                    ch = aa / float((y1 - y0) * (x1 - x0))
                else:
                    ch = 0.0
                r.update({f"{nm}_area": aa, f"{nm}_n_toc": len(lst), f"{nm}_rel": aa / float(aq), f"{nm}_iou": inter / float(aq + aa - inter), f"{nm}_cob_q": inter / float(aq),
                          f"{nm}_cob_alt": inter / float(max(aa, 1)), f"{nm}_cheio": ch, f"{nm}_nota": max((t[1] for t in lst), default=0.0),
                          f"{nm}_frase": max(lst, key=lambda t: t[1])[0] if lst else -1})
            da_inst.append(r)
            alts[(i, k)] = (q, U, Bm)
        for r in da_inst:
            for nm, col in (("par_q", "q_area"), ("u_par", "u_area"), ("b_par", "b_area")):
                v = [min(r[col], o[col]) / float(max(r[col], o[col])) for o in da_inst if o is not r and max(r[col], o[col]) > 0]
                r[nm] = max(v, default=0.0)
        rows += da_inst
    return rows, alts


def seleciona(base: list, shape: tuple, obj_pkl: str, n_caixas: int, mu, mb) -> list:
    """Troca cada pedaço pela alternativa de maior ganho estimado, se ele passa de LIMIAR (= planos_de + work)."""
    import pandas as pd
    rows, alts = pedacos(base, shape, obj_pkl, n_caixas)
    if not rows:
        return base
    P = pd.DataFrame(rows)
    pu = np.where(P.u_area > 0, mu.predict(_feats(P, "u")), -9.0)
    pb = np.where(P.b_area > 0, mb.predict(_feats(P, "b")), -9.0)
    esc = np.where(np.maximum(pu, pb) > LIMIAR, np.where(pu >= pb, 1, 2), 0)
    out = []
    for i, m in enumerate(base):
        novo = m.copy()
        for ii, k, e in zip(P.inst, P.k, esc):
            if ii == i and e:
                q, U, Bm = alts[(ii, k)]
                novo &= ~_gra(q, shape)
                novo |= _gra(U if e == 1 else Bm, shape)
        out.append(novo)
    return [m for m in out if m.any()]
