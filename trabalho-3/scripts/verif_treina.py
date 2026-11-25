#!/usr/bin/env python3
"""Verificador de candidatos: árvores com gradient boosting sobre as medidas de cada candidato (scripts/verif_cache.py),
para decidir se ele é cópia de verdade. Só a partição de treino.
Validação cruzada em 5 partes agrupadas por case_id: cada candidato recebe a probabilidade de um modelo que não viu
nenhuma imagem do seu caso. Com essas probabilidades, para cada corte, monta-se a resposta de cada figura (os
candidatos acima do corte, do mais provável ao menos, juntados como no fluxo) e mede-se a nota oficial.
Comparação: regras fixas aplicadas à MESMA tabela de candidatos.
Saída: <out>/candidatos.csv (medidas, rótulo y e probabilidade de fora da parte, p) e <out>/notas.csv.
Uso: python scripts/verif_treina.py --cache <dir de verif_cache.py> --sam-b <dir> --cortes 0.9 --out <dir>
"""
import argparse
import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cmfd import official, pipeline  # noqa: E402
from cmfd.detector import compose, passing  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="results/real/verif/cache")
ap.add_argument("--workers", type=int, default=31)
ap.add_argument("--min-f1", type=float, default=0.3, help="candidato conta como cópia de verdade se a máscara dele tem F1 >= isto com alguma fraude marcada")
ap.add_argument("--out", default="results/real/verif")
ap.add_argument("--sam-b", default="results/real/pares/sam")
ap.add_argument("--peso-compostas", type=float, default=1.0, help="peso no treino dos candidatos VERDADEIROS de figuras com várias fotos (são poucos)")
ap.add_argument("--peso-compostas-todos", type=float, default=1.0, help="peso no treino de TODOS os candidatos de figuras com várias fotos")
ap.add_argument("--aplica", default=None, help="tabela de candidatos de OUTRA partição (validação): o modelo treinado em toda a tabela de treino é aplicado nela")
ap.add_argument("--aplica-sam-b", default=None)
ap.add_argument("--rede", default=None, help="p_rede.npz de scripts/rede_treina.py: a nota da rede pequena entra como mais uma medida (mesma ordem da tabela)")
ap.add_argument("--rede-aplica", default=None, help="p_rede_aplica.npz, para a tabela de --aplica")
ap.add_argument("--extra", default=None, help="outra tabela de candidatos (régua sintética) que entra SÓ no treino de cada parte, nunca na avaliação")
ap.add_argument("--modelo", default="atual", choices=["atual", "maior", "parada", "lgbm", "xgb"], help="quem dá a nota aos candidatos (scripts/verif_modelos.py compara)")
ap.add_argument("--oraculo", type=float, default=0.0, help="TETO: em vez do modelo, a nota de cada candidato é 1 se o F1 dele com alguma fraude é >= isto (usa o gabarito; só para medir o limite da etapa de decisão)")
ap.add_argument("--cache-extra", default=None, help="tabela só com candidatos de outras fontes (verif_cache.py --so-fontes): entram junto com os de --cache, imagem a imagem")
ap.add_argument("--fontes-usa", default="", help="das fontes extras, só as de origem nesta lista (3 = KAZE, 4 = ORB, 5 = AKAZE); vazio = todas")
ap.add_argument("--aplica-extra", default=None, help="idem, para a tabela de --aplica")
ap.add_argument("--origens", default="", help="só os candidatos destas origens (ablação das passadas); vazio = todas")
ap.add_argument("--cortes", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
ap.add_argument("--sem", default="", help="medidas a deixar de fora, separadas por vírgula (ablação)")
a = ap.parse_args()
MEDIDAS = ["inliers", "inlier_ratio", "n_alive", "shift_rel", "spread_rel", "scale", "aniso", "reflect", "mirror_frac", "valid_geom",
           "src_area", "dst_area", "src_dst_iou", "resid_ncc", "s_margem", "s_frac", "s_pico", "s_lado_med", "cz", "tx_std", "tx_media",
           "tx_sat", "tx_grad", "area_px", "caixa_asp", "cheio", "borda", "vseed", "tile", "vseed_ncc", "origem",
           "mp", "n_cands", "pos_inliers", "rel_inliers", "tem_fotos", "tem_texto"]
CORTES = tuple(float(x) for x in a.cortes.split(","))
HOJE = replace(pipeline.SELECTION, vseed_min_inliers=3, vseed_min_margin=0.4, tile_min_inliers=8, tile_min_margin=0.4)
SUB10 = replace(pipeline.SELECTION, vseed_min_inliers=3, vseed_min_margin=0.4, tile_min_inliers=10 ** 9)


def faz_modelo():
    from sklearn.ensemble import HistGradientBoostingClassifier
    if a.modelo == "atual":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    if a.modelo == "maior":
        return HistGradientBoostingClassifier(max_iter=1200, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    if a.modelo == "parada":
        return HistGradientBoostingClassifier(max_iter=3000, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
                                              validation_fraction=0.15, n_iter_no_change=50, random_state=0)
    if a.modelo == "xgb":  # os mesmos valores de scripts/verif_modelos.py
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=16, random_state=0, tree_method="hist")
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          n_jobs=8, random_state=0, verbose=-1)


def carrega(arq, extra):
    """Lê a tabela de uma imagem e, se houver, acrescenta os candidatos das outras fontes (sempre na mesma ordem)."""
    d = pickle.load(open(arq, "rb"))
    usa = {float(v) for v in a.fontes_usa.split(",") if v}
    for pasta in (extra.split(",") if extra else []):   # várias pastas, separadas por vírgula, sempre na ordem dada
        e = os.path.join(pasta, os.path.basename(arq))
        assert os.path.exists(e), f"falta {e}"
        x = pickle.load(open(e, "rb"))
        if not d["pares"]:
            d["cands"] = list(d["cands"]) + [c for c in x["cands"] if not usa or c["stats"]["origem"] in usa]
    if a.origens:  # ablação das passadas de detecção: só os candidatos destas origens (0-2 = SIFT nas 3 escalas, 3 = KAZE, 4 = ORB, 5 = AKAZE)
        usa_o = {float(v) for v in a.origens.split(",")}
        d["cands"] = [c for c in d["cands"] if c["stats"]["origem"] in usa_o]
    return d


def abre(c, shape):
    out = []
    for b, p in ((c["bs"], c["ps"]), (c["bd"], c["pd"])):
        m = np.zeros(shape, bool)
        if b[1] > b[0]:
            h, w = b[1] - b[0], b[3] - b[2]
            m[b[0]:b[1], b[2]:b[3]] = np.unpackbits(np.frombuffer(p, np.uint8))[:h * w].reshape(h, w).astype(bool)
        out.append(m)
    return out


def nota(d, masks):
    ann = official.encode_prediction(masks)
    if d["gt"] == "authentic":
        return 1.0 if ann == "authentic" else 0.0
    return 0.0 if ann == "authentic" else official.score_one(d["gt"], ann, d["shape"])


def avalia(arg):
    arq, probs, extra = arg
    d = carrega(arq, extra)
    out = dict(path=d["path"], grupo=d["grupo"])
    if d["pares"]:
        out.update({k: d["nota_pares"] for k in ["hoje", "sub10"] + [f"v{c}{x}" for c in CORTES for x in ("", "|regras se 2+ fotos", "|regras se 4+ fotos", "|ou regras", "|ou regras se 2+ fotos")]})
        out["n_fotos"] = -1
        return out
    cs = d["cands"]
    for c in cs:
        c["src"], c["dst"] = abre(c, d["shape"])
    res = dict(cands=cs, shape=d["shape"])
    out["hoje"] = nota(d, compose(res, passing(res, HOJE), HOJE))
    out["sub10"] = nota(d, compose(res, passing(res, SUB10), SUB10))
    from cmfd.pairs import PairConfig, caixas
    B = np.load(os.path.join(a.sam_b, os.path.splitext(d["path"])[0].replace("/", "__") + ".npz"))["B"]
    nf = len(caixas(B, d["shape"][0], d["shape"][1], PairConfig(lado_min=0, sem_auto=True)))
    out["n_fotos"] = nf
    regra = list(passing(res, SUB10))
    for ct in CORTES:
        idx = [i for i in np.argsort(-probs) if probs[i] >= ct] if len(cs) else []
        out[f"v{ct}"] = nota(d, compose(res, tuple(idx), pipeline.SELECTION))
        # variações para as figuras com várias fotos
        out[f"v{ct}|regras se 2+ fotos"] = out["sub10"] if nf >= 2 else out[f"v{ct}"]
        out[f"v{ct}|regras se 4+ fotos"] = out["sub10"] if nf >= 4 else out[f"v{ct}"]
        ou = idx + [i for i in regra if i not in idx]
        out[f"v{ct}|ou regras"] = nota(d, compose(res, tuple(ou), pipeline.SELECTION))
        out[f"v{ct}|ou regras se 2+ fotos"] = out[f"v{ct}|ou regras"] if nf >= 2 else out[f"v{ct}"]
    return out


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    arqs = sorted(glob.glob(os.path.join(a.cache, "*.pkl")))
    rows, dono = [], []
    for k, arq in enumerate(arqs):
        d = carrega(arq, a.cache_extra)
        H, W = d["shape"]
        ins = sorted((c["stats"]["inliers"] for c in d["cands"]), reverse=True)
        for c in d["cands"]:
            st = c["stats"]
            r = {m: float(st.get(m, np.nan)) for m in MEDIDAS}
            r.update(shift_rel=st["shift"] / np.hypot(H, W), spread_rel=st.get("spread", 0.0) / np.hypot(H, W), mp=H * W / 1e6, n_cands=len(d["cands"]),
                     pos_inliers=ins.index(st["inliers"]), rel_inliers=st["inliers"] / max(ins[0], 1), tem_fotos=d["n_fotos"], tem_texto=float(d["tem_texto"]),
                     y=int(c["f1"] >= a.min_f1), f1=c["f1"], grupo=d["grupo"], caso=os.path.basename(d["path"]).split(".")[0])
            rows.append(r)
            dono.append(k)
    T = pd.DataFrame(rows)
    dono = np.asarray(dono)
    E = None
    if a.extra:
        er = []
        for arq in sorted(glob.glob(os.path.join(a.extra, "*.pkl"))):
            d = pickle.load(open(arq, "rb"))
            H, W = d["shape"]
            ins = sorted((c["stats"]["inliers"] for c in d["cands"]), reverse=True)
            for c in d["cands"]:
                st = c["stats"]
                r = {m: float(st.get(m, np.nan)) for m in MEDIDAS}
                r.update(shift_rel=st["shift"] / np.hypot(H, W), spread_rel=st.get("spread", 0.0) / np.hypot(H, W), mp=H * W / 1e6, n_cands=len(d["cands"]),
                         pos_inliers=ins.index(st["inliers"]), rel_inliers=st["inliers"] / max(ins[0], 1), tem_fotos=d["n_fotos"], tem_texto=float(d["tem_texto"]),
                         y=int(c["f1"] >= a.min_f1))
                er.append(r)
        E = pd.DataFrame(er)
        print("extra (só treino):", len(E), "candidatos,", int(E.y.sum()), "de cópia verdadeira")
    usa = [m for m in MEDIDAS if m not in a.sem.split(",")]
    if a.rede:
        from sklearn.metrics import roc_auc_score as _auc
        for i, arq_rede in enumerate(a.rede.split(",")):  # uma ou mais redes (arquivos separados por vírgula): cada nota vira uma medida
            z = np.load(arq_rede)
            assert len(z["p"]) == len(T) and list(z["n"]) == [int((dono == k).sum()) for k in range(len(arqs))], "a nota da rede não está na ordem desta tabela"
            col = "p_rede" if i == 0 else f"p_rede{i + 1}"
            T[col] = z["p"]
            usa.append(col)
            print(arq_rede, "sozinha: AUC", round(_auc(T.y, T[col]), 4))
    print(len(arqs), "imagens |", len(T), "candidatos |", int(T.y.sum()), "de cópia verdadeira |", "medidas:", len(usa))
    print(T.groupby("grupo").agg(cands=("y", "size"), verdadeiros=("y", "sum")).to_string())
    print("por fonte (origem: 0-2 = SIFT nas 3 escalas, 3 = KAZE, 4 = ORB, 5 = AKAZE):", T.groupby("origem").y.agg(["size", "sum"]).rename(columns={"size": "cands", "sum": "verdadeiros"}).T.to_dict())
    fz = T[T.grupo != "autenticas"].assign(nova=T.origem >= 3).groupby("caso").apply(lambda g: pd.Series(dict(sift=g.f1[~g.nova].max() if (~g.nova).any() else 0.0, nova=g.f1[g.nova].max() if g.nova.any() else 0.0)))
    if len(fz) and (T.origem >= 3).any():
        print(f"imagens com fraude e candidatos: {len(fz)} | sem candidato bom (F1 >= {a.min_f1}) do SIFT: {int((fz.sift < a.min_f1).sum())}, das quais as fontes novas dão um: {int(((fz.sift < a.min_f1) & (fz.nova >= a.min_f1)).sum())} | melhor F1 sobe > 0,1 em {int((fz.nova > fz.sift + 0.1).sum())}")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X, y = T[usa].to_numpy(np.float32), T.y.to_numpy()
    p = np.zeros(len(T))
    Xe, ye = (E[usa].to_numpy(np.float32), E.y.to_numpy()) if E is not None else (np.zeros((0, len(usa)), np.float32), np.zeros(0, int))
    comp = (T.grupo == "compostas").to_numpy()
    w = np.where(comp, a.peso_compostas_todos, 1.0) * np.where(comp & (y == 1), a.peso_compostas, 1.0)
    for tr, te in GroupKFold(5).split(X, y, T.caso):
        p[te] = faz_modelo().fit(
            np.vstack([X[tr], Xe]), np.r_[y[tr], ye], sample_weight=np.r_[w[tr], np.ones(len(ye))]).predict_proba(X[te])[:, 1]
    if a.oraculo:
        p = (T.f1.to_numpy() >= a.oraculo).astype(float)
        print("TETO: nota = gabarito (F1 >=", a.oraculo, ")")
    print("AUC fora da parte de treino:", round(roc_auc_score(y, p), 4))
    T.assign(p=p).to_csv(os.path.join(a.out, "candidatos.csv"), index=False)
    args = [(arq, p[dono == k], a.cache_extra) for k, arq in enumerate(arqs)]
    with ProcessPoolExecutor(a.workers) as ex:
        R = pd.DataFrame(list(ex.map(avalia, args, chunksize=4)))
    R.to_csv(os.path.join(a.out, "notas.csv"), index=False)
    cols = ["sub10", "hoje"] + [c for c in R.columns if c.startswith("v")]
    t = R.groupby("grupo")[cols].mean().T
    t["todas"] = R[cols].mean()
    t = t.rename(index={"sub10": "regras da submissão 10", "hoje": "regras + pedaços (margem 0,4)"})
    print(t.round(4).to_string())
    nf = R[R.n_fotos >= 0]
    print("figuras em que os pares não respondem, por nº de fotos do SAM:")
    print(nf.assign(fotos=nf.n_fotos.clip(upper=4)).groupby(["grupo", "fotos"]).size().unstack(fill_value=0).to_string())
    # importância por permutação, num modelo treinado em tudo (só para ler o que ele usa)
    from sklearn.inspection import permutation_importance
    m = faz_modelo().fit(X, y, sample_weight=w)
    imp = permutation_importance(m, X, y, n_repeats=3, random_state=0, scoring="roc_auc")
    print("medidas que mais pesam:", ", ".join(f"{usa[i]} {imp.importances_mean[i]:.3f}" for i in np.argsort(-imp.importances_mean)[:12]))
    if a.aplica:
        arqs2 = sorted(glob.glob(os.path.join(a.aplica, "*.pkl")))
        r2, dono2 = [], []
        for k, arq in enumerate(arqs2):
            d = carrega(arq, a.aplica_extra)
            H, W = d["shape"]
            ins = sorted((c["stats"]["inliers"] for c in d["cands"]), reverse=True)
            for c in d["cands"]:
                st = c["stats"]
                r = {mm: float(st.get(mm, np.nan)) for mm in MEDIDAS}
                r.update(shift_rel=st["shift"] / np.hypot(H, W), spread_rel=st.get("spread", 0.0) / np.hypot(H, W), mp=H * W / 1e6, n_cands=len(d["cands"]),
                         pos_inliers=ins.index(st["inliers"]), rel_inliers=st["inliers"] / max(ins[0], 1), tem_fotos=d["n_fotos"], tem_texto=float(d["tem_texto"]), y=int(c["f1"] >= a.min_f1), f1=c["f1"])
                r2.append(r)
                dono2.append(k)
        T2 = pd.DataFrame(r2)
        dono2 = np.asarray(dono2)
        if a.rede:
            assert len(a.rede_aplica.split(",")) == len(a.rede.split(",")), "--rede e --rede-aplica precisam ter o mesmo número de arquivos"
            for i, arq_rede in enumerate(a.rede_aplica.split(",")):
                z2 = np.load(arq_rede)
                assert len(z2["p"]) == len(T2), "a nota da rede não está na ordem da outra tabela"
                T2["p_rede" if i == 0 else f"p_rede{i + 1}"] = z2["p"]
        p2 = m.predict_proba(T2[usa].to_numpy(np.float32))[:, 1]
        if a.oraculo:
            p2 = (T2.f1.to_numpy() >= a.oraculo).astype(float)
        print("=== outra partição:", a.aplica, "|", len(arqs2), "imagens,", len(T2), "candidatos,", int(T2.y.sum()), "de cópia verdadeira | AUC", round(roc_auc_score(T2.y, p2), 4))
        T2.assign(p=p2, arq=[os.path.basename(arqs2[k]) for k in dono2]).to_csv(os.path.join(a.out, "candidatos_aplica.csv"), index=False)
        if a.aplica_sam_b:
            a.sam_b = a.aplica_sam_b
        with ProcessPoolExecutor(a.workers) as ex:
            R2 = pd.DataFrame(list(ex.map(avalia, [(arq, p2[dono2 == k], a.aplica_extra) for k, arq in enumerate(arqs2)], chunksize=4)))
        R2.to_csv(os.path.join(a.out, "notas_aplica.csv"), index=False)
        cols2 = ["sub10", "hoje"] + [c for c in R2.columns if c.startswith("v")]
        t2 = R2.groupby("grupo")[cols2].mean().T
        t2["todas"] = R2[cols2].mean()
        print(t2.rename(index={"sub10": "regras da submissão 10", "hoje": "regras + pedaços (margem 0,4)"}).round(4).to_string())
        print("n por grupo:", R2.groupby("grupo").size().to_dict())
    print("VERIF_FIM")
