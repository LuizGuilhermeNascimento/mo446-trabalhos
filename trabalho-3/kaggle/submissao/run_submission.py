"""Caderno da submissão ao Kaggle: o fluxo de cmfd/fluxo.py nas imagens de test_images.

Entradas: um Dataset com cmfd/, kaggle/, scripts/, metric/ e tabelas/ deste repositório, e a saída do caderno
kaggle/sam_pesos (pesos do SAM 3.1 e pacotes para instalar sem internet). Qualquer erro derruba o caderno.
"""
import os
import shutil
import subprocess
import sys
import time

EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
W = "/kaggle/working"


def find(name, root="/kaggle/input", depth=6):
    for d, sub, files in os.walk(root):
        if d[len(root):].count(os.sep) > depth:
            sub[:] = []
            continue
        if name in files or name in sub:
            return d
    return None


code = find("cmfd")
if code is None:
    raise SystemExit("ERRO: não achei o código (cmfd/) em /kaggle/input")
for f in ("kaggle/predict_fluxo.py", "kaggle/sam_paineis.py", "scripts/sam_objetos.py", "cmfd/fluxo.py", "tabelas/candidatos.csv", "tabelas/pedacos.csv"):
    if not os.path.exists(os.path.join(code, f)):
        raise SystemExit("ERRO: falta no Dataset de código: " + f)
comp = find("test_images")
n_cpu = len(os.sched_getaffinity(0))
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
if not gpus:
    raise SystemExit("ERRO: o caderno precisa de GPU")
rel = sorted("test_images/" + f for f in os.listdir(os.path.join(comp, "test_images")) if f.lower().endswith(EXT))
lista = os.path.join(W, "lista_imgs.txt")
open(lista, "w").write("\n".join(sorted(rel, key=lambda r: -os.path.getsize(os.path.join(comp, r)))) + "\n")  # as maiores primeiro
print("imagens:", len(rel), "| cpus:", n_cpu, "| GPUs:", gpus, flush=True)

wheels = find("wheels")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", os.path.join(wheels, "wheels"), "transformers==5.22.2"], check=True)
pesos = os.path.join(find("sam31"), "sam31")
off = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
py = sys.executable
sam_dir, obj_dir = os.path.join(W, "paineis"), os.path.join(W, "objetos")
aux = {  # fotos e texto na 1ª placa (junto com o SIFT); objetos na 2ª, se houver
    "SAM fotos/texto": subprocess.Popen([py, os.path.join(code, "kaggle", "sam_paineis.py"), "--root", comp, "--list", lista, "--model", pesos, "--out", sam_dir],
                                        env=dict(off, CUDA_VISIBLE_DEVICES="0")),
    "SAM objetos": subprocess.Popen([py, os.path.join(code, "scripts", "sam_objetos.py"), "--root", comp, "--list", lista, "--model", pesos, "--out", obj_dir],
                                    env=dict(off, CUDA_VISIBLE_DEVICES="1" if len(gpus) > 1 else "0")),
}
saida = os.path.join(W, "respostas_fluxo.csv")
pred = subprocess.Popen([py, os.path.join(code, "kaggle", "predict_fluxo.py"), "--root", comp, "--list", lista, "--sam", sam_dir, "--obj", obj_dir,
                         "--tab-ver", os.path.join(code, "tabelas", "candidatos.csv"), "--tab-sel", os.path.join(code, "tabelas", "pedacos.csv"),
                         "--workers", str(n_cpu), "--espera", "7200", "--out", saida],
                        env=dict(off, CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"))


def aborta(msg):
    for p in list(aux.values()) + [pred]:
        if p.poll() is None:
            p.kill()
    raise SystemExit("ERRO: " + msg)


while pred.poll() is None:
    for nm, p in aux.items():
        if p.poll() not in (None, 0):
            aborta(f"{nm} terminou com código {p.returncode}")
    time.sleep(5)
if pred.returncode != 0:
    aborta(f"o fluxo terminou com código {pred.returncode}")
for nm, p in aux.items():
    if p.wait() != 0:
        aborta(f"{nm} terminou com código {p.returncode}")

import pandas as pd  # noqa: E402
R = pd.read_csv(saida, dtype=str, keep_default_na=False)
S = pd.read_csv(os.path.join(comp, "sample_submission.csv"), dtype=str, keep_default_na=False)
assert len(R) == len(rel) and R.case_id.is_unique and set(S.case_id) == set(R.case_id), "as respostas não batem com o sample_submission"
assert (R.annotation.str.strip() != "").all(), "resposta vazia"
sub = S[["case_id"]].merge(R[["case_id", "annotation"]], on="case_id", how="left")
sub.to_csv(os.path.join(W, "submission.csv"), index=False)
print("linhas:", len(sub), "| com fraude:", int((sub.annotation != "authentic").sum()))
for d in (sam_dir, obj_dir):
    shutil.rmtree(d, ignore_errors=True)
for f in (saida + ".modelos.pkl", lista):
    os.remove(f)
print("SUBMISSAO_OK")
