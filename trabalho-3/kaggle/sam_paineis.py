#!/usr/bin/env python3
"""Fotos e texto de cada figura pelo SAM 3.1 congelado (pesos públicos): grava <out>/<nome>.npz por imagem.

Uso: python kaggle/sam_paineis.py --root <raiz> --list <arquivo com caminhos relativos> --model <pesos> --out <dir>
"""
import argparse
import os
import time

import numpy as np

FRASE, THR, LADO = "microscopic images", 0.3, 512
FRASES_TEXTO, THR_TEXTO, LADO_TEXTO, MAX_BOX_FRAC = ("text", "letters", "words", "numbers"), 0.3, 1024, 0.03
LOTE = 8


def main():
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rel = [l.strip() for l in open(a.list) if l.strip()]
    itens = [(os.path.join(a.root, r), os.path.splitext(r)[0].replace("/", "__")) for r in rel]
    itens.sort(key=lambda x: -os.path.getsize(x[0]))
    n_gpu = torch.cuda.device_count()
    dev = "cpu" if n_gpu == 0 else f"cuda:{n_gpu - 1}"
    dt = torch.float16 if n_gpu else torch.float32
    proc = Sam3Processor.from_pretrained(a.model, local_files_only=os.path.isdir(a.model))
    model = Sam3Model.from_pretrained(a.model, dtype=dt, local_files_only=os.path.isdir(a.model)).eval().to(dev)
    print(f"[sam] {dev} | {len(itens)} imagens | modelo em {time.time() - t0:.0f}s", flush=True)
    feitas, t1 = 0, time.time()
    for b in range(0, len(itens), LOTE):
        lote = itens[b:b + LOTE]
        imgs = [Image.open(p).convert("RGB") for p, _ in lote]
        alvo = []
        for im in imgs:
            s = LADO / max(im.size)
            alvo.append([max(1, round(im.size[1] * s)), max(1, round(im.size[0] * s))])
        inp = proc(images=imgs, text=[FRASE] * len(imgs), return_tensors="pt").to(dev)
        with torch.no_grad():
            ve = model.get_vision_features(pixel_values=inp["pixel_values"].to(dt))
            out = model(vision_embeds=ve, input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        res = proc.post_process_instance_segmentation(out, threshold=THR, mask_threshold=0.5, target_sizes=alvo)
        extra = [dict() for _ in lote]
        alvo_t = []
        for im in imgs:
            s = LADO_TEXTO / max(im.size)
            alvo_t.append([max(1, round(im.size[1] * s)), max(1, round(im.size[0] * s))])
        Ts, tn = [np.zeros(hw, bool) for hw in alvo_t], [0] * len(lote)
        for frase in FRASES_TEXTO:
            tk = proc(text=[frase] * len(imgs), return_tensors="pt").to(dev)
            with torch.no_grad():
                out_t = model(vision_embeds=ve, input_ids=tk["input_ids"], attention_mask=tk["attention_mask"])
            notas = out_t.pred_logits.sigmoid()
            if out_t.presence_logits is not None:
                notas = notas * out_t.presence_logits.sigmoid()
            bx = out_t.pred_boxes.float()
            peq = (notas > THR_TEXTO) & ((bx[..., 2] - bx[..., 0]) * (bx[..., 3] - bx[..., 1]) <= MAX_BOX_FRAC)
            for i, hw in enumerate(alvo_t):
                idx = torch.nonzero(peq[i]).flatten()
                tn[i] += int(len(idx))
                for k in range(0, len(idx), 16):
                    mk = out_t.pred_masks[i, idx[k:k + 16]].sigmoid().float()
                    mk = torch.nn.functional.interpolate(mk.unsqueeze(0), size=hw, mode="bilinear", align_corners=False)[0]
                    Ts[i] |= (mk > 0.5).any(0).cpu().numpy()
        for i, im in enumerate(imgs):
            extra[i] = dict(T=np.packbits(Ts[i]), thw=np.array(alvo_t[i]), tn=np.array(tn[i]), HW=np.array([im.size[1], im.size[0]]))
        for (p, cid), r, hw, ex in zip(lote, res, alvo, extra):
            m = r["masks"]
            P = m.any(0).cpu().numpy() if len(m) else np.zeros(hw, bool)
            B = np.zeros((len(m), 4), np.float32)
            for k, mk in enumerate(m.cpu().numpy().astype(bool)):
                ys, xs = np.nonzero(mk.any(1))[0], np.nonzero(mk.any(0))[0]
                if len(ys):
                    B[k] = (xs[0] / hw[1], ys[0] / hw[0], (xs[-1] + 1) / hw[1], (ys[-1] + 1) / hw[0])
            tmp = os.path.join(a.out, cid + ".tmp.npz")
            np.savez(tmp, P=np.packbits(P), hw=np.array(hw), n=np.array(len(m)), B=B, **ex)
            os.replace(tmp, os.path.join(a.out, cid + ".npz"))
        feitas += len(lote)
        if (b // LOTE) % 20 == 0 or feitas == len(itens):
            print(f"[sam] {feitas}/{len(itens)}  {(time.time() - t1) / feitas:.3f} s/img", flush=True)


if __name__ == "__main__":
    main()
