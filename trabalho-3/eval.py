#!/usr/bin/env python3
"""Nota oficial (oF1 da competição Recod.ai/LUC) de um arquivo de respostas numa das partições.

Uso:
    python eval.py --data <raiz dos dados> --split val --submission submission.csv
    python eval.py --data <raiz dos dados> --split val --baseline        # tudo "authentic", para conferir

O arquivo de respostas tem duas colunas, `path` e `annotation`, com uma linha por imagem da partição:
  - path: o caminho da imagem como está em splits/<partição>.csv (ex.: train_images/forged/10.png);
  - annotation: "authentic" ou as máscaras no formato da competição (metric/recodai_f1.py: rle_encode).
A chave é o caminho, e não o case_id, porque o mesmo case_id aparece como imagem autêntica e como forjada.

A nota de cada imagem sai da função oficial (metric/recodai_f1.py: score, sem modificações), e a nota da partição é
a média das imagens. Além do total, mostra a nota por grupo:
  - Sem fraude: imagens autênticas;
  - Várias fotos: figuras compostas com fraude (as de supplemental_images);
  - Únicas com fraude: as demais imagens com fraude (train_images/forged).
Qualquer problema no arquivo (imagem faltando, sobrando ou repetida) é erro: não existe resposta-padrão.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "metric"))
import recodai_f1 as oficial  # noqa: E402

GRUPOS = ["Sem fraude", "Várias fotos", "Únicas com fraude"]


def grupo(label: str, source: str) -> str:
    if label == "authentic":
        return "Sem fraude"
    return "Várias fotos" if source == "supplemental" else "Únicas com fraude"


def mascaras_gabarito(arq: str, hw: tuple) -> list:
    """Máscaras de um .npy da competição: (H, W), (n, H, W) ou (H, W, n)."""
    m = np.load(arq)
    if m.ndim == 2:
        ms = [m]
    elif m.ndim == 3:
        ms = [m[:, :, k] for k in range(m.shape[2])] if (m.shape[:2] == tuple(hw) and m.shape[2] != hw[0]) else [m[k] for k in range(m.shape[0])]
    else:
        raise ValueError(f"máscara com formato inesperado {m.shape} em {arq}")
    ms = [(x > 0).astype(np.uint8) for x in ms]
    for x in ms:
        if x.shape != tuple(hw):
            raise ValueError(f"máscara {x.shape} não tem o tamanho da imagem {hw} em {arq}")
    return [x for x in ms if x.any()]


def gabarito(data: str, sp: pd.DataFrame) -> pd.DataFrame:
    ann, shp = [], []
    for r in sp.itertuples():
        if r.label == "authentic":
            ann.append("authentic"); shp.append("authentic")
            continue
        with Image.open(os.path.join(data, r.path)) as im:
            hw = (im.height, im.width)
        ms = mascaras_gabarito(os.path.join(data, r.mask_path), hw)
        if not ms:
            raise ValueError(f"imagem com fraude sem máscara: {r.path}")
        ann.append(oficial.rle_encode(ms)); shp.append(json.dumps(list(hw)))
    return pd.DataFrame({"row_id": sp.path.to_numpy(), "annotation": ann, "shape": shp})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="pasta com train_images/, train_masks/, supplemental_images/ e supplemental_masks/")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--submission", help="csv com as colunas path e annotation")
    ap.add_argument("--baseline", action="store_true", help='avalia a resposta "authentic" em todas as imagens, sem arquivo')
    ap.add_argument("--por-imagem", help="grava um csv com a nota de cada imagem")
    a = ap.parse_args()
    if bool(a.submission) == a.baseline:
        sys.exit("ERRO: informe --submission <csv> ou --baseline (um dos dois)")

    sp = pd.read_csv(os.path.join(AQUI, "splits", a.split + ".csv"), dtype={"case_id": str}, keep_default_na=False)
    assert sp.path.is_unique
    if a.baseline:
        sub = pd.DataFrame({"path": sp.path, "annotation": "authentic"})
    else:
        sub = pd.read_csv(a.submission, dtype=str, keep_default_na=False)
        falta = {"path", "annotation"} - set(sub.columns)
        if falta:
            sys.exit(f"ERRO: faltam colunas no arquivo de respostas: {sorted(falta)}")
        if not sub.path.is_unique:
            sys.exit(f"ERRO: imagens repetidas no arquivo de respostas: {sub.path[sub.path.duplicated()].tolist()[:5]}")
        a_mais, a_menos = set(sub.path) - set(sp.path), set(sp.path) - set(sub.path)
        if a_mais or a_menos:
            sys.exit(f"ERRO: o arquivo não cobre a partição {a.split}: {len(a_menos)} imagens faltando (ex.: {sorted(a_menos)[:3]}), "
                     f"{len(a_mais)} que não são dela (ex.: {sorted(a_mais)[:3]})")
        if (sub.annotation.str.strip() == "").any():
            sys.exit("ERRO: há linhas com annotation vazia")
        sub = sub.set_index("path").loc[sp.path].reset_index()

    sol = gabarito(a.data, sp)
    pred = pd.DataFrame({"row_id": sub.path.to_numpy(), "annotation": sub.annotation.to_numpy()})
    notas = np.array([oficial.score(sol.iloc[[i]].reset_index(drop=True), pred.iloc[[i]].reset_index(drop=True), row_id_column_name="row_id")
                      for i in range(len(sp))])
    total = oficial.score(sol.copy(), pred.copy(), row_id_column_name="row_id")
    assert abs(total - notas.mean()) < 1e-9, (total, notas.mean())  # a nota oficial é a média das imagens

    R = pd.DataFrame({"path": sp.path, "grupo": [grupo(l, s) for l, s in zip(sp.label, sp.source)], "nota": notas,
                      "respondeu_fraude": pred.annotation.to_numpy() != "authentic"})
    print(f"Partição: {a.split} ({len(R)} imagens) | respostas: {'tudo authentic' if a.baseline else a.submission}\n")
    print(f"{'Grupo':<20}{'Imagens':>9}{'Nota':>9}{'Marcadas como fraude':>23}")
    for g in GRUPOS:
        x = R[R.grupo == g]
        if len(x):
            print(f"{g:<20}{len(x):>9}{x.nota.mean():>9.4f}{int(x.respondeu_fraude.sum()):>23}")
    print(f"{'Todas':<20}{len(R):>9}{total:>9.4f}{int(R.respondeu_fraude.sum()):>23}")
    if a.por_imagem:
        R.to_csv(a.por_imagem, index=False)
        print("\nnota por imagem em", a.por_imagem)


if __name__ == "__main__":
    main()
