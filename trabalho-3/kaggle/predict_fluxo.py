#!/usr/bin/env python3
"""Respostas do fluxo (cmfd/fluxo.py) para uma lista de imagens: csv com path, case_id e annotation.

Treina o verificador e o seletor a partir das tabelas do treino (candidatos.csv e pedacos.csv) e aplica imagem a
imagem. As fotos/texto (kaggle/sam_paineis.py) e os objetos (scripts/sam_objetos.py) do SAM podem estar sendo
gravados por outros processos: cada imagem espera o arquivo dela até --espera segundos; passou disso, é erro.
Qualquer erro numa imagem derruba tudo (não existe resposta-padrão).
Uso: python kaggle/predict_fluxo.py --root <raiz> --list <txt> --sam <dir> --obj <dir> --tab-ver <csv> --tab-sel <csv> --out <csv>
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--list", required=True, help="caminhos das imagens, relativos a --root")
ap.add_argument("--sam", required=True)
ap.add_argument("--obj", required=True)
ap.add_argument("--tab-ver", required=True)
ap.add_argument("--tab-sel", required=True)
ap.add_argument("--sift", default="kornia", choices=["kornia", "opencv"])
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--espera", type=float, default=0.0, help="segundos que cada imagem espera pelos arquivos do SAM (0 = já têm de existir)")
ap.add_argument("--modelos", default=None, help="onde guardar os modelos treinados (padrão: ao lado de --out)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
nome = lambda p: os.path.splitext(p)[0].replace("/", "__")
MOD = a.modelos or (a.out + ".modelos.pkl")
_M = None


def espera(arq):
    t0 = time.time()
    while not os.path.exists(arq):
        if time.time() - t0 > a.espera:
            raise FileNotFoundError(f"não apareceu em {a.espera:.0f} s: {arq}")
        time.sleep(2)
    return arq


def work(rel):
    global _M
    import cv2
    cv2.setNumThreads(1)
    from cmfd import fluxo, official
    from cmfd.data import load_gray
    if _M is None:
        _M = pickle.load(open(MOD, "rb"))
    t0 = time.time()
    full = load_gray(os.path.join(a.root, rel))
    d = fluxo.candidatos(full, espera(os.path.join(a.sam, nome(rel) + ".npz")), a.sift)
    info = dict(path=rel, case_id=os.path.splitext(os.path.basename(rel))[0], pares=bool(d["pares"]), n_cands=len(d["cands"]), n_aceitos=0, trocou=False)
    if d["pares"]:
        ms = d["pares"]
    else:
        probs = _M["ver"].predict_proba(fluxo.medidas(d))[:, 1] if d["cands"] else np.zeros(0)
        info["n_aceitos"] = int((probs >= fluxo.CORTE).sum())
        ms = fluxo.aceitos(d, probs)
        if ms:
            antes = [m.copy() for m in ms]
            ms = fluxo.seleciona(ms, d["shape"], espera(os.path.join(a.obj, nome(rel) + ".pkl")), d["n_caixas"], _M["mu"], _M["mb"])
            info["trocou"] = len(ms) != len(antes) or any((x != y).any() for x, y in zip(ms, antes))
    info.update(annotation=official.encode_prediction(ms), n_mascaras=len([m for m in ms if m.any()]), seg=time.time() - t0)
    return info


if __name__ == "__main__":
    import sklearn
    from cmfd import fluxo
    t0 = time.time()
    rel = [l.strip() for l in open(a.list) if l.strip()]
    assert len(set(rel)) == len(rel), "imagens repetidas na lista"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(MOD)), exist_ok=True)
    T, P = pd.read_csv(a.tab_ver), pd.read_csv(a.tab_sel)
    ver = fluxo.treina_verificador(T)
    mu, mb = fluxo.treina_seletor(P)
    pickle.dump(dict(ver=ver, mu=mu, mb=mb), open(MOD, "wb"))
    print(f"[fluxo] scikit-learn {sklearn.__version__} | verificador: {len(T)} candidatos de treino, {int(T.y.sum())} verdadeiros | seletor: {len(P)} pedaços | {time.time() - t0:.0f} s", flush=True)
    ordem = sorted(rel, key=lambda r: -os.path.getsize(os.path.join(a.root, r)))  # as maiores primeiro, como o SAM
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, ordem)):
            res.append(r)
            if (k + 1) % 50 == 0 or k + 1 == len(ordem):
                print(f"[fluxo] {k + 1}/{len(ordem)}  {(time.time() - t0) / (k + 1):.2f} s por imagem de relógio", flush=True)
    R = pd.DataFrame(res).set_index("path").loc[rel].reset_index()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    R[["path", "case_id", "annotation"]].to_csv(a.out, index=False)
    R.drop(columns="annotation").to_csv(a.out + ".info.csv", index=False)
    print(f"[fluxo] {len(R)} imagens | com fraude: {int((R.annotation != 'authentic').sum())} | pelas fotos: {int(R.pares.sum())} | seletor trocou em {int(R.trocou.sum())} | "
          f"{R.seg.mean():.1f} s por imagem por processo | total {time.time() - t0:.0f} s")
    print("FLUXO_FIM")
