#!/usr/bin/env python3
"""Seletor de máscara: mantém cada pedaço da máscara ou troca pelo objeto que o SAM recortou ali."""
import argparse
import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cmfd import official, pipeline  # noqa: E402
from cmfd.data import load_masks  # noqa: E402
from cmfd.detector import compose, passing  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--cache", default="results/real/verif/cache_tr")
ap.add_argument("--notas", default="results/real/verif/tr_full_rede_b/candidatos.csv")
ap.add_argument("--obj", default="results/real/sam_obj_train")
ap.add_argument("--sam-b", default="results/real/sam_train")
ap.add_argument("--split", default="splits/train.csv")
ap.add_argument("--aplica-cache", default=None)
ap.add_argument("--aplica-notas", default=None)
ap.add_argument("--aplica-obj", default=None)
ap.add_argument("--aplica-sam-b", default=None)
ap.add_argument("--aplica-split", default=None)
ap.add_argument("--cache-extra", default=None, help="candidatos de outras fontes (verif_cache.py --so-fontes), como em verif_treina.py")
ap.add_argument("--aplica-extra", default=None)
ap.add_argument("--corte", type=float, default=0.9)
ap.add_argument("--corte2", type=float, default=None, help="se algum candidato passa de --corte, entram também os de nota >= isto (a imagem já foi dada como fraudada; falta não perder a 2ª cópia)")
ap.add_argument("--sem-regras", action="store_true", help="só o verificador: não soma os candidatos das regras fixas nas figuras com 2+ fotos")
ap.add_argument("--origens", default="", help="só os candidatos destas origens (ablação das passadas); vazio = todas")
ap.add_argument("--max-frac", type=float, default=0.25, help="objeto maior que esta fração da imagem não conta (é a foto inteira)")
ap.add_argument("--toca", type=float, default=0.5)
ap.add_argument("--limiares", default="0,0.02,0.05,0.1,0.15")
ap.add_argument("--grava", default="0.02", help="limiar cujas máscaras finais são gravadas em <out>/mascaras_{treino,aplica} (para scripts/objpares.py)")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", default="results/real/verif/seletor")
a = ap.parse_args()
for v in (a.split, a.cache, a.obj, a.aplica_split or "", a.aplica_cache or "", a.aplica_obj or ""):
    assert "test" not in v, "a partição de teste só roda uma vez, no fim"
SUB10 = replace(pipeline.SELECTION, vseed_min_inliers=3, vseed_min_margin=0.4, tile_min_inliers=10 ** 9)
LIM = [float(x) for x in a.limiares.split(",")]
nome = lambda p: os.path.splitext(p)[0].replace("/", "__")
COMUNS = ["q_area", "q_lado", "q_cheio", "q_frac", "n_pecas", "n_inst", "mp", "n_obj", "par_q", "regras"]
DO_ALT = ["n_toc", "rel", "iou", "cob_q", "cob_alt", "cheio", "nota", "frase", "par"]


def abre(c, shape):
    out = []
    for b, p in ((c["bs"], c["ps"]), (c["bd"], c["pd"])):
        m = np.zeros(shape, bool)
        if b[1] > b[0]:
            h, w = b[1] - b[0], b[3] - b[2]
            m[b[0]:b[1], b[2]:b[3]] = np.unpackbits(np.frombuffer(p, np.uint8))[:h * w].reshape(h, w).astype(bool)
        out.append(m)
    return out


def carrega(arq, extra):
    d = pickle.load(open(arq, "rb"))
    for pasta in (extra.split(",") if extra else []):
        e = os.path.join(pasta, os.path.basename(arq))
        assert os.path.exists(e), f"falta {e}"
        if not d["pares"]:
            d["cands"] = list(d["cands"]) + list(pickle.load(open(e, "rb"))["cands"])
    if a.origens:  # ablação das passadas de detecção: só os candidatos destas origens (0-2 = SIFT nas 3 escalas, 3 = KAZE, 4 = ORB, 5 = AKAZE)
        usa_o = {float(v) for v in a.origens.split(",")}
        d["cands"] = [c for c in d["cands"] if c["stats"]["origem"] in usa_o]
    return d


def nota(d, masks):
    masks = [m for m in masks if m.any()]
    ann = official.encode_prediction(masks)
    if d["gt"] == "authentic":
        return 1.0 if ann == "authentic" else 0.0
    return 0.0 if ann == "authentic" else official.score_one(d["gt"], ann, d["shape"])


def pequeno(m, hw):
    return m if m.shape == tuple(hw) else cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def grande(m, shape):
    return m if m.shape == tuple(shape) else cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def caixa_de(m):
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def iou_max(x, pecas_gt):
    """Maior IoU de x com algum pedaço de fraude marcada."""
    if not x.any():
        return 0.0
    y0, y1, x0, x1 = caixa_de(x)
    ax, best = int(x.sum()), 0.0
    for g, ag in pecas_gt:
        inter = int((g[y0:y1, x0:x1] & x[y0:y1, x0:x1]).sum())
        if inter:
            best = max(best, inter / float(ax + ag - inter))
    return best


def work(job):
    arq, probs, mask_path, obj_dir, sam_b, grava, extra, planos = job
    cv2.setNumThreads(1)
    d = carrega(arq, extra)
    shape = tuple(d["shape"])
    out = dict(path=d["path"], grupo=d["grupo"], caso=os.path.basename(d["path"]).split(".")[0], fixo=True)
    passo2 = planos is not None

    def guarda(ms):
        if passo2:
            pickle.dump(dict(path=d["path"], shape=shape, pares=bool(d["pares"]), bits=[np.packbits(m).tobytes() for m in ms]), open(os.path.join(grava, nome(d["path"]) + ".pkl"), "wb"))
    if d["pares"]:
        out["base"] = d["nota_pares"]
        guarda([])
        return out, []
    cs = d["cands"]
    for c in cs:
        c["src"], c["dst"] = abre(c, shape)
    res = dict(cands=cs, shape=shape)
    from cmfd.pairs import PairConfig, caixas
    B = np.load(os.path.join(sam_b, nome(d["path"]) + ".npz"))["B"]
    nf = len(caixas(B, shape[0], shape[1], PairConfig(lado_min=0, sem_auto=True)))
    idx = [i for i in np.argsort(-probs) if probs[i] >= a.corte] if len(cs) else []
    if idx and a.corte2 is not None:
        idx = [i for i in np.argsort(-probs) if probs[i] >= a.corte2]
    if nf >= 2 and not a.sem_regras:  # variante com regras fixas nas figuras de 2+ fotos (não usada no fluxo final: --sem-regras)
        idx += [i for i in passing(res, SUB10) if i not in idx]
    base = compose(res, tuple(idx), pipeline.SELECTION)
    base = [m.astype(bool) for m in base if m.any()]
    if planos is None:
        out["base"] = nota(d, base)
    if d["gt"] == "authentic" or not base:
        guarda(base)
        return out, []
    out["fixo"] = False
    # objetos do SAM, recortados, na resolução em que foram guardados
    z = pickle.load(open(os.path.join(obj_dir, nome(d["path"]) + ".pkl"), "rb"))
    hw = tuple(z["hw"])
    objs = []
    for o in z["objs"]:
        y0, y1, x0, x1 = o["caixa"]
        cr = np.unpackbits(np.frombuffer(o["bits"], np.uint8))[:(y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)
        ar = int(cr.sum())
        if 0 < ar <= a.max_frac * hw[0] * hw[1]:
            objs.append((o["frase"], o["nota"], (y0, y1, x0, x1), cr, ar))
    pecas_gt = []
    if planos is None:
        for g in load_masks(os.path.join(a.data, mask_path), shape):
            g = pequeno(np.asarray(g).astype(bool), hw)
            n, lab, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
            pecas_gt += [(lab == k, int(st[k, cv2.CC_STAT_AREA])) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 4]
    rows, alts = [], {}
    for i, m in enumerate(base):
        ms = pequeno(m, hw)
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
                if inter >= a.toca * min(ar, aq):
                    toc.append((f, nt, (y0, y1, x0, x1), cr))
            U = np.zeros(hw, bool)
            for _, _, (y0, y1, x0, x1), cr in toc:
                U[y0:y1, x0:x1] |= cr
            Bm = np.zeros(hw, bool)
            if melhor is not None:
                y0, y1, x0, x1 = melhor[3]
                Bm[y0:y1, x0:x1] = melhor[4]
            r = dict(path=d["path"], caso=out["caso"], inst=i, k=k, q_area=aq, q_lado=min(w, h), q_cheio=aq / float(w * h), q_frac=aq / float(hw[0] * hw[1]),
                     n_pecas=len(ks), n_inst=len(base), mp=shape[0] * shape[1] / 1e6, n_obj=len(objs), regras=int(nf >= 2))
            for nm, alt, lst in (("u", U, toc), ("b", Bm, [melhor[1:]] if melhor else [])):
                aa = int(alt.sum())
                inter = int((alt & q).sum())
                if aa:
                    y0, y1, x0, x1 = caixa_de(alt)
                    ch = aa / float((y1 - y0) * (x1 - x0))
                else:
                    ch = 0.0
                r.update({f"{nm}_area": aa, f"{nm}_n_toc": len(lst), f"{nm}_rel": aa / float(aq), f"{nm}_iou": inter / float(aq + aa - inter), f"{nm}_cob_q": inter / float(aq),
                          f"{nm}_cob_alt": inter / float(max(aa, 1)), f"{nm}_cheio": ch, f"{nm}_nota": max((t[1] for t in lst), default=0.0),
                          f"{nm}_frase": max(lst, key=lambda t: t[1])[0] if lst else -1})
            if planos is None:
                r.update(g_q=iou_max(q, pecas_gt), g_u=iou_max(U, pecas_gt), g_b=iou_max(Bm, pecas_gt))
            da_inst.append(r)
            alts[(i, k)] = (q, U, Bm)
        for r in da_inst:  # os dois lados de uma cópia são o mesmo objeto: as alternativas dos pedaços da mesma máscara concordam em tamanho?
            for nm, col in (("par_q", "q_area"), ("u_par", "u_area"), ("b_par", "b_area")):
                v = [min(r[col], o[col]) / float(max(r[col], o[col])) for o in da_inst if o is not r and max(r[col], o[col]) > 0]
                r[nm] = max(v, default=0.0)
        rows += da_inst
    if planos is None:
        planos = dict(troca_u={(r["inst"], r["k"]): 1 for r in rows if r["u_area"]}, troca_b={(r["inst"], r["k"]): 2 for r in rows if r["b_area"]},
                      teto={(r["inst"], r["k"]): int(np.argmax([r["g_q"] + 1e-9, r["g_u"], r["g_b"]])) for r in rows})
    for nm, plano in planos.items():
        ms_ = []
        for i, m in enumerate(base):
            novo = m.copy()
            for (ii, k), esc in plano.items():
                if ii == i and esc:
                    q, U, Bm = alts[(ii, k)]
                    novo &= ~grande(q, shape)
                    novo |= grande(U if esc == 1 else Bm, shape)
            ms_.append(novo)
        out[nm] = nota(d, ms_)
        if nm == f"sel_{float(a.grava)}":
            guarda(ms_)
    if passo2 and f"sel_{float(a.grava)}" not in planos:
        guarda(base)
    return out, rows


def tabela(cache, notas, split, obj, sam_b, grava, extra=None):
    os.makedirs(grava, exist_ok=True)
    sp = pd.read_csv(split, dtype={"case_id": str}, keep_default_na=False).set_index("path")
    T = pd.read_csv(notas, usecols=["p"])
    arqs = sorted(glob.glob(os.path.join(cache, "*.pkl")))
    cab = [carrega(q, extra) for q in arqs]
    tam = [len(d["cands"]) for d in cab]
    assert sum(tam) == len(T), ("a tabela de notas não é desta tabela de candidatos", sum(tam), len(T))
    falta = [d["path"] for d in cab if not os.path.exists(os.path.join(obj, nome(d["path"]) + ".pkl"))]
    assert not falta, f"faltam objetos do SAM de {len(falta)} imagens, p. ex. {falta[:3]}"
    ini = np.r_[0, np.cumsum(tam)]
    return [(q, T.p.to_numpy()[ini[k]:ini[k + 1]], sp.loc[cab[k]["path"], "mask_path"], obj, sam_b, grava, extra) for k, q in enumerate(arqs)]


def roda(jobs, planos=None):
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, [j + (None if planos is None else planos.get(j[0], {}),) for j in jobs], chunksize=4))
    return pd.DataFrame([o for o, _ in res]), pd.DataFrame([r for _, rs in res for r in rs])


def feats(P, nm):
    return P[COMUNS + [f"{nm}_{c}" for c in DO_ALT]].to_numpy(np.float32)


def modelo():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


def planos_de(P, pu, pb, jobs):
    """Para cada limiar: a alternativa de maior ganho estimado, se passa do limiar (e se existe)."""
    pu = np.where(P.u_area > 0, pu, -9.0)
    pb = np.where(P.b_area > 0, pb, -9.0)
    por_path = {os.path.splitext(os.path.basename(j[0]))[0]: j[0] for j in jobs}
    planos = {}
    for t in LIM:
        esc = np.where(np.maximum(pu, pb) > t, np.where(pu >= pb, 1, 2), 0)
        for path, i, k, e in zip(P.path, P.inst, P.k, esc):
            planos.setdefault(por_path[nome(path)], {}).setdefault(f"sel_{t}", {})[(i, k)] = int(e)
    return planos


def resumo(I, titulo):
    cols = [c for c in ("base", "troca_u", "troca_b", "teto") + tuple(f"sel_{t}" for t in LIM) if c in I.columns]
    J = I.copy()
    for c in cols[1:]:
        J[c] = J[c].where(~J.fixo, J.base)
    t = J.groupby("grupo")[cols].mean().T
    t["todas"] = J[cols].mean()
    print(f"\n=== {titulo}: {len(J)} imagens | com máscara a decidir: {int((~J.fixo).sum())}")
    print(t.round(4).to_string())
    print("n por grupo:", J.groupby("grupo").size().to_dict())
    M = J[~J.fixo]
    for c in cols[1:]:
        print(f"  {c}: melhora em {int((M[c] > M.base + 1e-6).sum())}, piora em {int((M[c] < M.base - 1e-6).sum())} | nessas {len(M)} imagens: {M.base.mean():.4f} -> {M[c].mean():.4f}")
    return J


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    from sklearn.model_selection import GroupKFold
    jobs = tabela(a.cache, a.notas, a.split, a.obj, a.sam_b, os.path.join(a.out, "mascaras_treino"), a.cache_extra)
    I, P = roda(jobs)
    print(len(jobs), "imagens |", len(P), "pedaços de máscara em", P.path.nunique(), "imagens | com alternativa U:", int((P.u_area > 0).sum()), "| com B:", int((P.b_area > 0).sum()), flush=True)
    print(f"IoU médio com a fraude marcada: pedaço atual {P.g_q.mean():.3f} | U {P.g_u.mean():.3f} | B {P.g_b.mean():.3f} | o melhor dos três {P[['g_q', 'g_u', 'g_b']].max(1).mean():.3f}")
    pu, pb = np.zeros(len(P)), np.zeros(len(P))
    for tr, te in GroupKFold(5).split(P, groups=P.caso):
        pu[te] = modelo().fit(feats(P.iloc[tr], "u"), (P.g_u - P.g_q).to_numpy()[tr]).predict(feats(P.iloc[te], "u"))
        pb[te] = modelo().fit(feats(P.iloc[tr], "b"), (P.g_b - P.g_q).to_numpy()[tr]).predict(feats(P.iloc[te], "b"))
    print("ganho real x estimado (correlação): U", round(float(np.corrcoef(pu, P.g_u - P.g_q)[0, 1]), 3), "| B", round(float(np.corrcoef(pb, P.g_b - P.g_q)[0, 1]), 3))
    P.assign(pu=pu, pb=pb).to_csv(os.path.join(a.out, "pedacos.csv"), index=False)
    I2, _ = roda(jobs, planos_de(P, pu, pb, jobs))
    I = I.merge(I2[["path"] + [c for c in I2.columns if c.startswith("sel_")]], on="path", how="left")
    resumo(I, "TREINO (seletor fora da parte de treino)").to_csv(os.path.join(a.out, "imagens.csv"), index=False)
    if a.aplica_cache:
        mu = modelo().fit(feats(P, "u"), (P.g_u - P.g_q).to_numpy())
        mb = modelo().fit(feats(P, "b"), (P.g_b - P.g_q).to_numpy())
        from sklearn.inspection import permutation_importance
        nomes = COMUNS + DO_ALT
        imp = permutation_importance(mb, feats(P, "b"), (P.g_b - P.g_q).to_numpy(), n_repeats=3, random_state=0)
        print("\nmedidas que mais pesam (B):", ", ".join(f"{nomes[i]} {imp.importances_mean[i]:.3f}" for i in np.argsort(-imp.importances_mean)[:8]))
        jobs2 = tabela(a.aplica_cache, a.aplica_notas, a.aplica_split, a.aplica_obj, a.aplica_sam_b, os.path.join(a.out, "mascaras_aplica"), a.aplica_extra)
        V, PV = roda(jobs2)
        V2, _ = roda(jobs2, planos_de(PV, mu.predict(feats(PV, "u")), mb.predict(feats(PV, "b")), jobs2))
        V = V.merge(V2[["path"] + [c for c in V2.columns if c.startswith("sel_")]], on="path", how="left")
        resumo(V, "VALIDAÇÃO (seletor treinado em todo o treino)").to_csv(os.path.join(a.out, "imagens_aplica.csv"), index=False)
    print("SELETOR_FIM")
