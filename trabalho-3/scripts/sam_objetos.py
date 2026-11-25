#!/usr/bin/env python3
"""Objetos de dentro das fotos (células, manchas, bandas) pelo SAM 3.1 congelado, pesos públicos.

Servem ao seletor de máscara: a máscara de um objeto inteiro pode substituir o pedaço achado pela comparação de
pixels. A ideia de recortar o conteúdo com o SAM vem de soluções públicas da competição (3º e 15º lugares). Para cada
imagem e cada frase, guarda cada objeto: nota, caixa e máscara (empacotada dentro da caixa), no tamanho da imagem
(lado maior limitado a --lado). O codificador de visão roda uma vez por imagem; cada frase só refaz a parte leve.
Saída: <out>/<nome>.pkl = dict(hw=(h, w) da máscara, HW=(H, W) da imagem, objs=[dict(frase, nota, caixa=(y0, y1, x0, x1), bits)]).
Uso: python scripts/sam_objetos.py --root <raiz> --list <arquivo de caminhos> --model <pesos> --out <dir> [--parte i/n]
"""
import argparse
import os
import pickle
import time

import numpy as np

FRASES = ("biological cells", "scientific blobs", "item of interest", "western blot band", "spot")  # as 3 primeiras são as do 3º lugar
THR, MAX_OBJ = 0.3, 400

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lado", type=int, default=1536, help="lado maior das máscaras guardadas (a imagem menor que isso fica no tamanho dela)")
ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"])
ap.add_argument("--parte", default="0/1")
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
assert "test" not in os.path.basename(a.list), "a partição de teste só roda uma vez, no fim"

if __name__ == "__main__":
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    os.makedirs(a.out, exist_ok=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    rel = [l.strip() for l in open(a.list) if l.strip()]
    pi, pn = map(int, a.parte.split("/"))
    rel = rel[pi::pn]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if (a.dtype == "fp16" and dev == "cuda") else torch.float32
    proc = Sam3Processor.from_pretrained(a.model, local_files_only=os.path.isdir(a.model))
    model = Sam3Model.from_pretrained(a.model, dtype=dt, local_files_only=os.path.isdir(a.model)).eval().to(dev)
    print(f"[objetos] {dev} | {a.dtype} | {len(rel)} imagens | frases: {FRASES}", flush=True)
    t0 = time.time()
    for k, r in enumerate(rel):
        arq = os.path.join(a.out, os.path.splitext(r)[0].replace("/", "__") + ".pkl")
        if os.path.exists(arq):
            continue
        im = Image.open(os.path.join(a.root, r)).convert("RGB")
        W, H = im.size
        s = min(1.0, a.lado / max(H, W))
        hw = [max(1, round(H * s)), max(1, round(W * s))]
        inp = proc(images=[im], text=[FRASES[0]], return_tensors="pt").to(dev)
        objs = []
        with torch.no_grad():
            ve = model.get_vision_features(pixel_values=inp["pixel_values"].to(dt))
            for f, frase in enumerate(FRASES):
                tk = proc(text=[frase], return_tensors="pt").to(dev)
                out = model(vision_embeds=ve, input_ids=tk["input_ids"], attention_mask=tk["attention_mask"])
                notas = out.pred_logits.sigmoid()[0]
                if out.presence_logits is not None:
                    notas = notas * out.presence_logits.sigmoid()[0]
                idx = torch.nonzero(notas > THR).flatten()
                idx = idx[torch.argsort(notas[idx], descending=True)][:MAX_OBJ]
                for i0 in range(0, len(idx), 16):  # em blocos: a memória não cresce com o nº de objetos
                    ii = idx[i0:i0 + 16]
                    mk = torch.nn.functional.interpolate(out.pred_masks[0, ii].sigmoid().float().unsqueeze(0), size=hw, mode="bilinear", align_corners=False)[0] > 0.5
                    for j, m in zip(ii.tolist(), mk.cpu().numpy()):
                        ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
                        if not len(ys):
                            continue
                        y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
                        objs.append(dict(frase=f, nota=float(notas[j]), caixa=(y0, y1, x0, x1), bits=np.packbits(m[y0:y1, x0:x1]).tobytes()))
        pickle.dump(dict(path=r, hw=tuple(hw), HW=(H, W), objs=objs), open(arq + ".tmp", "wb"))
        os.replace(arq + ".tmp", arq)
        if (k + 1) % 10 == 0 or k + 1 == len(rel):
            print(f"[objetos] {k + 1}/{len(rel)}  {(time.time() - t0) / (k + 1):.1f} s/img  último: {len(objs)} objetos", flush=True)
    print("OBJ_FIM", flush=True)
