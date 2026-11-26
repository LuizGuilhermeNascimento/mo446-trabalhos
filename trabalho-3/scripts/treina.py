#!/usr/bin/env python3
"""Treino, só com a partição de treino: gera as tabelas do verificador e do seletor de máscara.

Uso: python scripts/treina.py --data <raiz> --sam <dir> --obj <dir> --work <dir> [--workers 4] [--out tabelas]
"""
import argparse
import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cmfd import fluxo  # noqa: E402
from cmfd.data import load_gray, load_masks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="splits/train.csv")
ap.add_argument("--sam", required=True, help="saída de kaggle/sam_paineis.py")
ap.add_argument("--obj", required=True, help="saída de scripts/sam_objetos.py")
ap.add_argument("--work", required=True)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--out", default="tabelas")
a = ap.parse_args()
MIN_F1 = 0.3  # candidato conta como cópia de verdade se a máscara dele tem F1 >= isto com alguma fraude marcada
nome = lambda p: os.path.splitext(p)[0].replace("/", "__")


def empacota(m):
    ys, xs = np.nonzero(m)
    if not len(ys):
        return (0, 0, 0, 0), b""
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return (y0, y1, x0, x1), np.packbits(m[y0:y1, x0:x1]).tobytes()


def abre(pacote, shape):
    (y0, y1, x0, x1), bits = pacote
    m = np.zeros(shape, bool)
    if y1 > y0:
        m[y0:y1, x0:x1] = np.unpackbits(np.frombuffer(bits, np.uint8))[:(y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)
    return m


def candidatos_de(r):
    """Candidatos de uma imagem, com as medidas e o quanto cada um acerta a fraude marcada (retomável)."""
    cv2.setNumThreads(1)
    arq = os.path.join(a.work, nome(r["path"]) + ".pkl")
    if os.path.exists(arq):
        return
    full = load_gray(os.path.join(a.data, r["path"]))
    d = fluxo.candidatos(full, os.path.join(a.sam, nome(r["path"]) + ".npz"))
    gts = [np.asarray(m).astype(bool) for m in load_masks(os.path.join(a.data, r["mask_path"]), full.shape)] if r["mask_path"] else []
    cands = []
    for c in d["cands"]:
        m = c["src"] | c["dst"]
        f1 = 0.0
        for g in gts:
            inter = float((m & g).sum())
            if inter:
                f1 = max(f1, 2 * inter / (m.sum() + g.sum()))
        cands.append(dict(stats=c["stats"], f1=f1, src=empacota(c["src"]), dst=empacota(c["dst"])))
    out = dict(d, path=r["path"], mask_path=r["mask_path"], pares=bool(d["pares"]), cands=cands, com_fraude=any(g.any() for g in gts))
    pickle.dump(out, open(arq + ".tmp", "wb"))
    os.replace(arq + ".tmp", arq)


def iou_max(x, pecas):
    if not x.any():
        return 0.0
    y0, y1, x0, x1 = fluxo._caixa(x)
    ax, best = int(x.sum()), 0.0
    for g, ag in pecas:
        inter = int((g[y0:y1, x0:x1] & x[y0:y1, x0:x1]).sum())
        if inter:
            best = max(best, inter / float(ax + ag - inter))
    return best


def pedacos_de(job):
    """Pedaços das máscaras aceitas de uma imagem com fraude, com o IoU real de cada alternativa."""
    cv2.setNumThreads(1)
    arq, probs = job
    d = pickle.load(open(arq, "rb"))
    if d["pares"] or not d["com_fraude"]:
        return []
    shape = tuple(d["shape"])
    for c in d["cands"]:
        c["src"], c["dst"] = abre(c["src"], shape), abre(c["dst"], shape)
    base = fluxo.aceitos(d, probs)
    if not base:
        return []
    rows, alts = fluxo.pedacos(base, shape, os.path.join(a.obj, nome(d["path"]) + ".pkl"), d["n_caixas"])
    if not rows:
        return []
    hw = next(iter(alts.values()))[0].shape
    pecas = []
    for g in load_masks(os.path.join(a.data, d["mask_path"]), shape):
        n, lab, st, _ = cv2.connectedComponentsWithStats(fluxo._peq(np.asarray(g).astype(bool), hw).astype(np.uint8), connectivity=8)
        pecas += [(lab == k, int(st[k, cv2.CC_STAT_AREA])) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 4]
    for r in rows:
        q, U, Bm = alts[(r["inst"], r["k"])]
        r.update(g_q=iou_max(q, pecas), g_u=iou_max(U, pecas), g_b=iou_max(Bm, pecas))
    return rows


if __name__ == "__main__":
    from sklearn.model_selection import GroupKFold
    assert "test" not in os.path.basename(a.split), "o treino não usa a partição de teste"
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    sp = pd.read_csv(a.split, dtype={"case_id": str}, keep_default_na=False)
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(candidatos_de, sp.to_dict("records")))
    arqs = sorted(os.path.join(a.work, nome(p) + ".pkl") for p in sp.path)
    rows, tam = [], []
    for arq in arqs:
        d = pickle.load(open(arq, "rb"))
        caso = os.path.basename(d["path"]).split(".")[0]
        rows += [dict(r, y=int(c["f1"] >= MIN_F1), caso=caso) for r, c in zip(fluxo.linhas(d), d["cands"])]
        tam.append(len(d["cands"]))
    T = pd.DataFrame(rows)
    X, y = T[fluxo.MEDIDAS].to_numpy(np.float32), T.y.to_numpy()
    p = np.zeros(len(T))
    for tr, te in GroupKFold(5).split(X, y, T.caso):  # cada candidato recebe a probabilidade de um modelo que não viu o caso dele
        p[te] = fluxo.verificador().fit(X[tr], y[tr], sample_weight=np.ones(len(tr))).predict_proba(X[te])[:, 1]
    T[fluxo.MEDIDAS + ["y"]].to_csv(os.path.join(a.out, "candidatos.csv"), index=False)
    ini = np.r_[0, np.cumsum(tam)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(pedacos_de, [(arq, p[ini[k]:ini[k + 1]]) for k, arq in enumerate(arqs)], chunksize=4))
    P = pd.DataFrame([r for rs in res for r in rs])
    P = P[fluxo.COMUNS + [f"{n}_{c}" for n in "ub" for c in fluxo.DO_ALT] + ["u_area", "b_area", "g_q", "g_u", "g_b"]]
    P.to_csv(os.path.join(a.out, "pedacos.csv"), index=False)
    print(f"{len(arqs)} imagens | {len(T)} candidatos, {int(T.y.sum())} verdadeiros | {len(P)} pedaços")
    print("TREINO_OK")
