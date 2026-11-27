#!/usr/bin/env python3
"""Objetos de dentro das fotos (células, manchas, bandas) pelo SAM 3.1 congelado: grava <out>/<nome>.pkl por imagem.

Uso: python scripts/sam_objetos.py --root <raiz> --list <arquivo de caminhos> --model <pesos> --out <dir>
"""
import argparse
import os
import pickle
import time

import numpy as np

FRASES = ("biological cells", "scientific blobs", "item of interest", "western blot band", "spot")
THR, MAX_OBJ, LADO = 0.3, 400, 1536  # LADO: lado maior das máscaras guardadas

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

if __name__ == "__main__":
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    os.makedirs(a.out, exist_ok=True)
    rel = [l.strip() for l in open(a.list) if l.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor.from_pretrained(a.model, local_files_only=os.path.isdir(a.model))
    model = Sam3Model.from_pretrained(a.model, dtype=torch.float32, local_files_only=os.path.isdir(a.model)).eval().to(dev)
    print(f"[objetos] {dev} | {len(rel)} imagens", flush=True)
    t0 = time.time()
    for k, r in enumerate(rel):
        arq = os.path.join(a.out, os.path.splitext(r)[0].replace("/", "__") + ".pkl")
        if os.path.exists(arq):
            continue
        im = Image.open(os.path.join(a.root, r)).convert("RGB")
        W, H = im.size
        s = min(1.0, LADO / max(H, W))
        hw = [max(1, round(H * s)), max(1, round(W * s))]
        inp = proc(images=[im], text=[FRASES[0]], return_tensors="pt").to(dev)
        objs = []
        with torch.no_grad():
            ve = model.get_vision_features(pixel_values=inp["pixel_values"].to(torch.float32))
            for f, frase in enumerate(FRASES):
                tk = proc(text=[frase], return_tensors="pt").to(dev)
                out = model(vision_embeds=ve, input_ids=tk["input_ids"], attention_mask=tk["attention_mask"])
                notas = out.pred_logits.sigmoid()[0]
                if out.presence_logits is not None:
                    notas = notas * out.presence_logits.sigmoid()[0]
                idx = torch.nonzero(notas > THR).flatten()
                idx = idx[torch.argsort(notas[idx], descending=True)][:MAX_OBJ]
                for i0 in range(0, len(idx), 16):
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
