"""Guarda, para uso SEM internet, os pesos públicos do SAM 3.1 (cópia unsloth/sam3.1) em /kaggle/working/sam31 e os
pacotes do transformers em /kaggle/working/wheels.

Não toca em dados da competição nem faz submissão. No fim, recarrega do disco em fp16 e roda numa imagem sintética.
"""
import os
import subprocess
import sys
import time

T0 = time.time()
W = "/kaggle/working/wheels"  # pacotes para instalar sem internet
subprocess.run([sys.executable, "-m", "pip", "download", "-q", "transformers==5.22.2", "-d", W], check=True)
print("wheels:", sorted(os.listdir(W)), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", W, "transformers==5.22.2"], check=True)
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from transformers import Sam3Model, Sam3Processor  # noqa: E402

D = "/kaggle/working/sam31"
proc = Sam3Processor.from_pretrained("unsloth/sam3.1")
model = Sam3Model.from_pretrained("unsloth/sam3.1")
proc.save_pretrained(D)
model.save_pretrained(D)
del proc, model
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
proc = Sam3Processor.from_pretrained(D, local_files_only=True)
model = Sam3Model.from_pretrained(D, dtype=dt, local_files_only=True).eval().to(dev)
img = Image.new("RGB", (900, 600), "white")
dr = ImageDraw.Draw(img)
for x, y in ((40, 40), (480, 40), (40, 320), (480, 320)):
    dr.rectangle([x, y, x + 380, y + 240], fill=(90, 90, 90), outline="black")
    for k in range(40):
        dr.ellipse([x + 9 * k, y + (37 * k) % 220, x + 9 * k + 14, y + (37 * k) % 220 + 14], fill=(200, 200, 200))
inp = proc(images=[img] * 8, text=["microscopic images"] * 8, return_tensors="pt").to(dev)
inp["pixel_values"] = inp["pixel_values"].to(dt)
t = time.time()
with torch.no_grad():
    out = model(**inp)
res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=[[341, 512]] * 8)
print("instâncias por imagem:", [len(r["masks"]) for r in res], "| lote de 8 em", round(time.time() - t, 2), "s |", dev,
      "| pico", round(torch.cuda.max_memory_allocated() / 2**30, 1) if dev == "cuda" else 0, "GB")
print(D, sorted((f, os.path.getsize(os.path.join(D, f)) // 1024) for f in os.listdir(D)))
print(f"PESOS_PRONTO total {time.time() - T0:.0f}s")
