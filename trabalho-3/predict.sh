#!/usr/bin/env bash
# Predições de uma partição. Uso: DATA=<dados da competição> ./predict.sh [train|val|test] [saida.csv]
# Opcionais: SAM, WORK, WORKERS, PYTHON. Precisa de GPU e de tabelas/ (saída do train.sh).
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA:?defina DATA com a pasta dos dados da competição}"
PART=${1:-test}; SAIDA=${2:-predicoes_teste.csv}
SAM=${SAM:-unsloth/sam3.1}; WORK=${WORK:-work}; WORKERS=${WORKERS:-4}; PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="$WORK/$PART"; mkdir -p "$P"
tail -n +2 "splits/$PART.csv" | cut -d, -f2 > "$P/lista.txt"

$PY kaggle/sam_paineis.py --root "$DATA" --list "$P/lista.txt" --model "$SAM" --out "$P/sam"
$PY scripts/sam_objetos.py --root "$DATA" --list "$P/lista.txt" --model "$SAM" --out "$P/obj"
$PY kaggle/predict_fluxo.py --root "$DATA" --list "$P/lista.txt" --sam "$P/sam" --obj "$P/obj" \
    --tab-ver tabelas/candidatos.csv --tab-sel tabelas/pedacos.csv --workers "$WORKERS" --out "$P/respostas.csv"
$PY -c "
import sys, pandas as pd
R = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
R[['case_id', 'annotation', 'path']].to_csv(sys.argv[2], index=False)
print(sys.argv[2], len(R), 'imagens,', int((R.annotation != 'authentic').sum()), 'marcadas como fraude')" "$P/respostas.csv" "$SAIDA"
