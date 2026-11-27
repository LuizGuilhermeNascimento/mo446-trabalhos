#!/usr/bin/env python3
"""Baseline clássico: pontos SIFT casados com a própria imagem, teste de razão, RANSAC e máscara pelos pontos.

Uso:
    python baseline_sift.py --data <raiz> --split val --varre 4,6,8,10,15,20,30,50   # S para cada mínimo de pares
    python baseline_sift.py --data <raiz> --split test --min-pares 10 --out baseline_teste.csv
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from cmfd import official  # noqa: E402
from cmfd.data import load_gray  # noqa: E402
from cmfd.ksift import sift_kornia  # noqa: E402

LADO, MAX_PONTOS, RAZAO, DIST_MIN, ERRO_PX, MIN_BRUTO = 2048, 8000, 0.6, 20.0, 3.0, 4


def pares(pts, des, dev):
    """Para cada ponto, o mais parecido fora da vizinhança dele, se passar no teste de razão."""
    P, D = torch.from_numpy(pts).to(dev), torch.from_numpy(des).to(dev)
    d = torch.cdist(D, D)
    d[torch.cdist(P, P) < DIST_MIN] = float("inf")
    v, i = d.topk(2, largest=False)
    ok = v[:, 0] < RAZAO * v[:, 1]
    a, b = torch.nonzero(ok).flatten().cpu().numpy(), i[ok, 0].cpu().numpy()
    return np.unique(np.sort(np.stack([a, b], 1), 1), axis=0)


def detecta(gray, dev):
    """Devolve (nº de pares concordantes, máscara ou None)."""
    H, W = gray.shape
    s = min(1.0, LADO / max(H, W))
    peq = cv2.resize(gray, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    pts, des = sift_kornia(peq, MAX_PONTOS)
    if len(pts) < 4:
        return 0, None
    ij = pares(pts, des, dev)
    if len(ij) < MIN_BRUTO:
        return 0, None
    a, b = pts[ij[:, 0]], pts[ij[:, 1]]
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=ERRO_PX)
    if M is None or int(inl.sum()) < MIN_BRUTO:
        return 0, None
    k = inl.ravel().astype(bool)
    m = np.zeros((H, W), np.uint8)
    for lado in (a[k], b[k]):
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(lado / s).astype(np.int32)), 1)
    return int(k.sum()), cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--min-pares", type=int, default=None, help="mínimo de pares concordantes para responder fraude")
    ap.add_argument("--varre", default="", help="lista de mínimos: mostra S de cada um (para escolher na validação)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.varre and a.split == "test":
        sys.exit("ERRO: o mínimo de pares é escolhido na validação, não no teste")
    if not a.varre and (a.min_pares is None or not a.out):
        sys.exit("ERRO: informe --varre <lista> ou --min-pares <n> --out <csv>")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = pd.read_csv(os.path.join(AQUI, "splits", a.split + ".csv"), dtype={"case_id": str}, keep_default_na=False)
    n, ann = [], []
    for k, p in enumerate(sp.path):
        ni, m = detecta(load_gray(os.path.join(a.data, p)), dev)
        n.append(ni)
        ann.append(official.encode_prediction([m]) if m is not None else "authentic")
        if (k + 1) % 100 == 0 or k + 1 == len(sp):
            print(f"[baseline] {k + 1}/{len(sp)}", flush=True)
    n = np.array(n)
    if a.varre:
        import eval as ev
        sol = ev.gabarito(a.data, sp)
        nota = np.array([1.0 if (g == "authentic" and r == "authentic") else 0.0 if "authentic" in (g, r) else
                         official.score_one(g, r, tuple(json.loads(h))) for g, r, h in zip(sol.annotation, ann, sol["shape"])])
        vazio = (sol.annotation == "authentic").to_numpy().astype(float)
        print(f"{'mínimo de pares':>16}{'S':>9}{'marcadas como fraude':>23}")
        for t in (int(x) for x in a.varre.split(",")):
            print(f"{t:>16}{np.where(n >= t, nota, vazio).mean():>9.4f}{int((n >= t).sum()):>23}")
    if a.out:
        R = pd.DataFrame({"case_id": sp.case_id, "annotation": np.where(n >= a.min_pares, ann, "authentic"), "path": sp.path, "pares": n})
        R.to_csv(a.out, index=False)
        print(f"{a.out}: {len(R)} imagens, {int((R.annotation != 'authentic').sum())} marcadas como fraude (mínimo de {a.min_pares} pares)")


if __name__ == "__main__":
    main()
