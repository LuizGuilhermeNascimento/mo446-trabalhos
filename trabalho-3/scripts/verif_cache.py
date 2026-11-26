#!/usr/bin/env python3
"""Tabela de candidatos para o verificador: roda a procura de cópias com os cortes ABERTOS nas imagens de uma partição."""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cmfd import official, pipeline  # noqa: E402
from cmfd.data import load_gray, load_masks  # noqa: E402
from cmfd.detector import detect_candidates  # noqa: E402
from cmfd.pairs import PairConfig, pair_masks  # noqa: E402
from cmfd.redepatch import recortes  # noqa: E402
from cmfd.panelfilter import load_panels  # noqa: E402
from cmfd.textfilter import load_sam_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--sam", required=True)
ap.add_argument("--sam-b", required=True)
ap.add_argument("--workers", type=int, default=31)
ap.add_argument("--out", default="results/real/verif/cache")
ap.add_argument("--split", default=None, help="csv de uma partição (ex.: splits/val.csv) no lugar da amostra de treino; o SAM dela vem de --sam e --sam-b")
ap.add_argument("--manifesto", default=None, help="csv com path, mask_path, label e tipo (régua sintética) no lugar da amostra de treino")
ap.add_argument("--sift", default="opencv", choices=["opencv", "kornia"], help="kornia: SIFT de GPU na procura dentro da figura (a comparação entre fotos segue no OpenCV, como na submissão 10)")
ap.add_argument("--fontes", default="", help="detectores extras, separados por vírgula (kaze, akaze, orb): cada um roda a procura dentro da figura na maior escala e acrescenta os seus candidatos (medida origem = 3, 4, ...)")
ap.add_argument("--fontes-escala", type=int, default=0, choices=[0, 1, 2], help="em qual das 3 ampliações (pipeline.SCALES) rodam as fontes extras; 0 = 2048 (a de sempre)")
ap.add_argument("--so-fontes", action="store_true", help="grava só os candidatos dos detectores extras (para juntar a uma tabela já pronta: verif_treina.py --cache-extra)")
ap.add_argument("--sem-pares", action="store_true", help="não deixa a comparação entre fotos responder: guarda sempre os candidatos de dentro da figura")
a = ap.parse_args()
nome = lambda p: os.path.splitext(p)[0].replace("/", "__")
# cortes abertos: o que os filtros fixos cortavam passa a ser decidido pelo verificador
ABERTO = replace(pipeline.SELECTION, min_resid_ncc=-1.0, max_aniso=2.0, max_src_dst_iou=0.3, min_rel_inliers=0.0,
                 vseed_min_inliers=2, vseed_min_margin=-1.0, tile_min_inliers=0, tile_min_margin=-1.0, pico_todos=True, dedup_iou=1.0)


def recorta(m):
    ys, xs = np.nonzero(m)
    if not len(ys):
        return (0, 0, 0, 0), b""
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (int(y0), int(y1), int(x0), int(x1)), np.packbits(m[y0:y1, x0:x1]).tobytes()


def work(r):
    cv2.setNumThreads(1)
    arq = os.path.join(a.out, nome(r["path"]) + ".pkl")
    if os.path.exists(arq):
        return 0
    for v in pipeline.SCALES:
        pipeline.VARIANTS[v] = replace(pipeline.VARIANTS[v], clahe=2.0, mirror=True, vlinks_k=2, vlinks_thr=0.8, vlinks_max=8, vlinks_radius=4,
                                       tiles=True, tile_ratio=0.75, tile_min_pairs=4, tile_max=16, sift_impl=a.sift)
    t0 = time.time()
    full = load_gray(os.path.join(a.data, r["path"]))
    H, W = full.shape
    f = os.path.join(a.sam, nome(r["path"]) + ".npz")
    P, T = load_panels(f), load_sam_text(f)
    gts = [np.asarray(m).astype(bool) for m in load_masks(os.path.join(a.data, r["mask_path"]), full.shape)] if r["mask_path"] else []
    gt_ann = official.encode_prediction(gts) if gts else "authentic"
    out = dict(path=r["path"], label=r["label"], grupo=r["grupo"], shape=(H, W), gt=gt_ann, pares=False, nota_pares=None, cands=[],
               n_fotos=0 if P is None else 1, tem_texto=T is not None)
    ms = [] if a.sem_pares else pair_masks(full, np.load(os.path.join(a.sam_b, nome(r["path"]) + ".npz"))["B"], PairConfig(lado_min=0, sem_auto=True))[0]
    if ms:
        ann = official.encode_prediction(ms)
        out.update(pares=True, nota_pares=(0.0 if not gts else official.score_one(gt_ann, ann, full.shape)))
    else:
        rfull = None
        ORIG = {"kaze": 3 + 3 * a.fontes_escala, "orb": 4 + 3 * a.fontes_escala, "akaze": 5 + 3 * a.fontes_escala}  # escala 1: origens 6, 7, 8; escala 2: 9, 10, 11
        passes = [] if a.so_fontes else [(vi, v, a.sift) for vi, v in enumerate(pipeline.SCALES)]
        passes += [(ORIG[f], pipeline.SCALES[a.fontes_escala], f) for f in a.fontes.split(",") if f]
        for vi, v, impl in passes:
            res = detect_candidates(full, replace(pipeline.VARIANTS[v], matcher=pipeline.MATCHER, sift_impl=impl), masks=False)
            part, rfull = pipeline.remask(full, res, v, rfull, ABERTO, T, P)
            for c in part:
                m = c["src"] | c["dst"]
                if not m.any():
                    continue
                st = dict(c["stats"], origem=float(vi))
                f1 = prec = rev = 0.0
                for g in gts:
                    inter = float((m & g).sum())
                    if inter:
                        f = 2 * inter / (m.sum() + g.sum())
                        if f > f1:
                            f1, prec, rev = f, inter / m.sum(), inter / g.sum()
                bs, ps = recorta(c["src"])
                bd, pd_ = recorta(c["dst"])
                img, rui = recortes(full, rfull, c["An"], c["src"])  # para a rede pequena (cmfd/redepatch.py)
                out["cands"].append(dict(stats=st, f1=f1, prec=prec, rev=rev, bs=bs, ps=ps, bd=bd, pd=pd_, An=np.asarray(c["An"], np.float32), img=img, rui=rui))
    out["seg"] = time.time() - t0
    pickle.dump(out, open(arq + ".tmp", "wb"))
    os.replace(arq + ".tmp", arq)
    return len(out["cands"])


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    if a.split:
        assert "test" not in os.path.basename(a.split), "a partição de teste só roda uma vez, no fim"
        ref = pd.read_csv(a.split, dtype={"case_id": str}, keep_default_na=False)
        ref["grupo"] = np.where(ref.label == "authentic", "autenticas", np.where(ref.source == "supplemental", "compostas", "forjadas_comuns"))
    elif a.manifesto:  # régua sintética (feita só de imagens sem fraude do treino)
        ref = pd.read_csv(a.manifesto, keep_default_na=False)
        ref["grupo"] = "sint_" + ref.tipo
    else:
        ref = pd.read_csv("results/real/sam3_texto/por_imagem.csv", keep_default_na=False)
        sp = pd.read_csv("splits/train.csv", dtype={"case_id": str}, keep_default_na=False)  # só treino
        ref = ref.merge(sp[["path", "mask_path"]], on="path")  # por_imagem.csv já traz label e grupo
        assert len(ref) == 671, len(ref)
    with ProcessPoolExecutor(a.workers) as p:
        n = list(p.map(work, ref.to_dict("records")))
    print(len(n), "imagens,", sum(n), "candidatos novos gravados em", a.out)
    print("CACHE_FIM")
