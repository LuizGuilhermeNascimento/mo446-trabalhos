#!/usr/bin/env python3
"""Recria as partições de splits/ (70/15/15, semente 42) a partir dos dados da competição.

Todas as imagens de um mesmo case_id (a autêntica e a forjada) ficam na mesma partição, e o sorteio é feito dentro de
cada tipo de caso (só autêntica, só forjada, as duas; treino ou suplementar), para as três partições terem a mesma
composição. Os caminhos ficam relativos à raiz dos dados.
Uso: python scripts/make_splits.py --data <raiz dos dados> [--out splits]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for source in ("train", "supplemental"):
        img_dir, mdir = root / f"{source}_images", root / f"{source}_masks"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() not in EXT:
                continue
            mpath = mdir / f"{p.stem}.npy"
            pai = p.parent.name.lower()
            label = pai if pai in ("authentic", "forged") else ("forged" if mpath.exists() else "authentic")
            rows.append(dict(case_id=p.stem, path=str(p.relative_to(root)), label=label, source=source,
                             mask_path=str(mpath.relative_to(root)) if (mpath.exists() and label == "forged") else ""))
    if not rows:
        raise FileNotFoundError(f"nenhuma imagem encontrada em {root}")
    return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame, seed: int = 42, fracs=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    g = df.groupby("case_id").agg(labels=("label", lambda s: "+".join(sorted(set(s)))), source=("source", "first")).reset_index()
    g["stratum"] = g["labels"] + "|" + g["source"]
    rng = np.random.default_rng(seed)
    assign = {}
    for _, sub in g.groupby("stratum"):
        ids = sub["case_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_tr, n_va = int(round(fracs[0] * len(ids))), int(round(fracs[1] * len(ids)))
        for k, cid in enumerate(ids):
            assign[cid] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    out = df.copy()
    out["split"] = out["case_id"].map(assign)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits"))
    a = ap.parse_args()
    sp = make_splits(scan(Path(a.data).resolve()), seed=a.seed)
    assert sp.groupby("case_id")["split"].nunique().max() == 1  # nenhum caso em duas partições
    os.makedirs(a.out, exist_ok=True)
    for s in ("train", "val", "test"):
        x = sp[sp.split == s].drop(columns="split")
        x.to_csv(os.path.join(a.out, f"{s}.csv"), index=False)
        print(f"{s}: {len(x)} imagens, {x.case_id.nunique()} casos")
