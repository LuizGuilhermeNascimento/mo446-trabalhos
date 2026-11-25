#!/usr/bin/env python3
"""Fotos (painéis) e texto de cada figura pelo SAM 3.1 congelado, pesos públicos.

Processa as imagens em lotes, das maiores para as menores (a mesma ordem de kaggle/predict_fluxo.py). Para cada
imagem grava <saida>/<nome>.npz: P = união das instâncias da frase "microscopic images" (bits, reduzida a 512 px no
lado maior), n = nº de instâncias, B = caixa de cada foto. Com --texto, grava também T = texto da figura (frases
"text", "letters", "words", "numbers"; lado maior 1024 px), para o filtro de texto. No fim grava _FIM; se o modelo
não carregar, grava _FALHA. Qualquer falha é erro (não existe resposta-padrão).
Uso: python kaggle/sam_paineis.py --root <raiz> --list <arquivo com caminhos relativos> --model <pesos> --out <dir> --texto
     (ou --images <dir> [<dir> ...]; com --list, o nome do arquivo é o caminho com '/' trocado por '__')
"""
import argparse
import json
import os
import time

import numpy as np

EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
FRASE, THR, LADO = "microscopic images", 0.3, 512
# --texto: na mesma passada (o codificador de visão roda uma vez só), o texto da figura
FRASES_TEXTO, THR_TEXTO, LADO_TEXTO, MAX_BOX_FRAC = ("text", "letters", "words", "numbers"), 0.3, 1024, 0.03


def grava_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--root", default=None)
    ap.add_argument("--list", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--texto", action="store_true", help="grava também a máscara de texto (T, thw, tn, HW)")
    ap.add_argument("--parte", default="0/1", help="i/n: só as imagens de índice i módulo n (medição no treino)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.list:
        rel = [l.strip() for l in open(a.list) if l.strip()]
        itens = [(os.path.join(a.root, r), os.path.splitext(r)[0].replace("/", "__")) for r in rel]
    else:
        ps = [os.path.join(d, f) for d in a.images for f in sorted(os.listdir(d)) if f.lower().endswith(EXT)]
        itens = [(p, os.path.splitext(os.path.basename(p))[0]) for p in ps]
    itens.sort(key=lambda x: -os.path.getsize(x[0]))  # a mesma ordem de predict_fluxo.py
    if a.n:
        itens = itens[:a.n]
    pi, pn = map(int, a.parte.split("/"))
    itens = itens[pi::pn]
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import Sam3Model, Sam3Processor
        if a.threads:
            torch.set_num_threads(a.threads)
        n_gpu = torch.cuda.device_count()
        dev = "cpu" if n_gpu == 0 else f"cuda:{n_gpu - 1}"  # com 2 GPUs, a busca de vizinhos fica na 0 e o SAM na 1
        dt = torch.float16 if (a.dtype == "fp16" and n_gpu) else torch.float32
        proc = Sam3Processor.from_pretrained(a.model, local_files_only=os.path.isdir(a.model))
        model = Sam3Model.from_pretrained(a.model, dtype=dt, local_files_only=os.path.isdir(a.model)).eval().to(dev)
        print(f"[sam] transformers {transformers.__version__} | {dev} | {a.dtype} | lote {a.batch} | {len(itens)} imagens | "
              f"modelo em {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        grava_json(os.path.join(a.out, "_FALHA"), dict(erro=repr(e)[:500]))
        print("[sam] FALHA ao carregar:", repr(e)[:500], flush=True)
        raise
    feitas, t1 = 0, time.time()
    for b in range(0, len(itens), a.batch):
        lote = itens[b:b + a.batch]
        imgs = [Image.open(p).convert("RGB") for p, _ in lote]
        alvo = []
        for im in imgs:
            s = LADO / max(im.size)
            alvo.append([max(1, round(im.size[1] * s)), max(1, round(im.size[0] * s))])
        inp = proc(images=imgs, text=[FRASE] * len(imgs), return_tensors="pt").to(dev)
        with torch.no_grad():
            ve = model.get_vision_features(pixel_values=inp["pixel_values"].to(dt))  # uma vez por lote; as frases reaproveitam
            out = model(vision_embeds=ve, input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        res = proc.post_process_instance_segmentation(out, threshold=THR, mask_threshold=0.5, target_sizes=alvo)
        extra = [dict() for _ in lote]
        if a.texto:
            alvo_t = []
            for im in imgs:
                s = LADO_TEXTO / max(im.size)
                alvo_t.append([max(1, round(im.size[1] * s)), max(1, round(im.size[0] * s))])
            Ts, tn = [np.zeros(hw, bool) for hw in alvo_t], [0] * len(lote)
            for frase in FRASES_TEXTO:
                tk = proc(text=[frase] * len(imgs), return_tensors="pt").to(dev)
                with torch.no_grad():
                    out_t = model(vision_embeds=ve, input_ids=tk["input_ids"], attention_mask=tk["attention_mask"])
                # mesma conta de post_process_instance_segmentation (nota = logit x presença, nota > limiar, máscara
                # ampliada e cortada em 0,5), mas em blocos de 16 instâncias e já unindo: a memória não cresce com
                # o nº de pedaços de texto (o original guarda cada instância como uma máscara 1024 x 1024 em int64)
                notas = out_t.pred_logits.sigmoid()
                if out_t.presence_logits is not None:
                    notas = notas * out_t.presence_logits.sigmoid()
                bx = out_t.pred_boxes.float()  # xyxy em fração da imagem
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
            B = np.zeros((len(m), 4), np.float32)  # caixa de cada painel (x0, y0, x1, y1) em fração da figura
            for k, mk in enumerate(m.cpu().numpy().astype(bool)):
                ys, xs = np.nonzero(mk.any(1))[0], np.nonzero(mk.any(0))[0]
                if len(ys):
                    B[k] = (xs[0] / hw[1], ys[0] / hw[0], (xs[-1] + 1) / hw[1], (ys[-1] + 1) / hw[0])
            tmp = os.path.join(a.out, cid + ".tmp.npz")
            np.savez(tmp, P=np.packbits(P), hw=np.array(hw), n=np.array(len(m)), B=B, **ex)
            os.replace(tmp, os.path.join(a.out, cid + ".npz"))
        feitas += len(lote)
        if (b // a.batch) % 20 == 0 or feitas == len(itens):
            mem = torch.cuda.max_memory_allocated(dev) / 2**30 if n_gpu else 0
            print(f"[sam] {feitas}/{len(itens)}  {(time.time() - t1) / feitas:.3f} s/img  pico {mem:.1f} GB  total {time.time() - t0:.0f}s",
                  flush=True)
    mem = torch.cuda.max_memory_allocated(dev) / 2**30 if n_gpu else 0
    grava_json(os.path.join(a.out, "_FIM"), dict(n=feitas, s_img=(time.time() - t1) / max(feitas, 1), pico_gb=mem, total_s=time.time() - t0))
    print(f"[sam] FIM {feitas} imagens, {(time.time() - t1) / max(feitas, 1):.3f} s/img, pico {mem:.1f} GB")


if __name__ == "__main__":
    main()
