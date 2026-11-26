#!/usr/bin/env python3
"""Tabelas de treino do fluxo final, só com as colunas que os modelos usam."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cmfd.fluxo import COMUNS, DO_ALT, MEDIDAS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ver", required=True)
ap.add_argument("--sel", required=True)
ap.add_argument("--out", default="tabelas")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T = pd.read_csv(a.ver)[MEDIDAS + ["y"]]
P = pd.read_csv(a.sel)
P = P[COMUNS + [f"{n}_{c}" for n in "ub" for c in DO_ALT] + ["u_area", "b_area", "g_q", "g_u", "g_b"]]
T.to_csv(os.path.join(a.out, "candidatos.csv"), index=False)
P.to_csv(os.path.join(a.out, "pedacos.csv"), index=False)
print(f"{a.out}/candidatos.csv: {len(T)} candidatos, {int(T.y.sum())} verdadeiros | {a.out}/pedacos.csv: {len(P)} pedaços")
