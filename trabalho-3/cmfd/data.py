"""Leitura do dataset da competição Recod.ai/LUC e criação das partições."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class Item:
    case_id: str
    path: str
    label: str          # "authentic" ou "forged"
    source: str         # "train" ou "supplemental"
    mask_path: str | None


def _mask_dir_for(root: Path, source: str) -> Path | None:
    for name in (f"{source}_masks", "train_masks" if source == "train" else "supplemental_masks"):
        d = root / name
        if d.is_dir():
            return d
    return None


def scan(root: str | os.PathLike) -> pd.DataFrame:
    """Lista todas as imagens rotuladas (treino + suplementares)."""
    root = Path(root)
    rows = []
    for source, img_dir in (("train", root / "train_images"), ("supplemental", root / "supplemental_images")):
        if not img_dir.is_dir():
            continue
        mdir = _mask_dir_for(root, source)
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            parent = p.parent.name.lower()
            cid = p.stem
            mpath = mdir / f"{cid}.npy" if mdir is not None else None
            has_mask = mpath is not None and mpath.exists()
            if parent in ("authentic", "forged"):
                label = parent
            else:  # suplementares podem vir sem subpasta: classe pela existência de máscara
                label = "forged" if has_mask else "authentic"
            rows.append(dict(case_id=cid, path=str(p), label=label, source=source,
                             mask_path=str(mpath) if (has_mask and label == "forged") else ""))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"nenhuma imagem encontrada em {root}")
    return df


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"não consegui ler {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def key_of(case_id: str, label: str, source: str) -> str:
    """Chave única de uma imagem no cache: 4 case_ids existem em train_images E em supplemental_images."""
    return f"{case_id}_{label[0]}" + ("s" if source == "supplemental" else "")


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df["key"] = [key_of(c, l, s) for c, l, s in zip(df["case_id"], df["label"], df["source"])]
    return df


def load_masks(mask_path: str, shape: tuple[int, int] | None = None) -> list[np.ndarray]:
    """Devolve a lista de máscaras binárias (uma por região) de um .npy."""
    if not mask_path:
        return []
    m = np.load(mask_path)
    if m.ndim == 2:
        ms = [m]
    elif m.ndim == 3:
        if shape is not None and m.shape[:2] == tuple(shape) and m.shape[2] != shape[0]:
            ms = [m[:, :, k] for k in range(m.shape[2])]
        else:
            ms = [m[k] for k in range(m.shape[0])]
    else:
        raise ValueError(f"máscara com formato inesperado {m.shape} em {mask_path}")
    return [(x > 0).astype(np.uint8) for x in ms]


def make_splits(df: pd.DataFrame, seed: int = 42, fracs=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """70/15/15 agrupado por case_id e estratificado pelo tipo do grupo."""
    g = df.groupby("case_id").agg(labels=("label", lambda s: "+".join(sorted(set(s)))),
                                  source=("source", "first")).reset_index()
    g["stratum"] = g["labels"] + "|" + g["source"]
    rng = np.random.default_rng(seed)
    assign = {}
    for _, sub in g.groupby("stratum"):
        ids = sub["case_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        for k, cid in enumerate(ids):
            assign[cid] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    out = df.copy()
    out["split"] = out["case_id"].map(assign)
    return out
