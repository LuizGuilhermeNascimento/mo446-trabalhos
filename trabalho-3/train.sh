#!/usr/bin/env bash
# Treino, só com a partição de treino. Uso: DATA=<dados da competição> ./train.sh
# Opcionais: SAM, WORK, WORKERS, PYTHON. Precisa de GPU. Saída: tabelas/candidatos.csv e tabelas/pedacos.csv.
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA:?defina DATA com a pasta dos dados da competição}"
SAM=${SAM:-unsloth/sam3.1}; WORK=${WORK:-work}; WORKERS=${WORKERS:-4}; PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
T="$WORK/train"; mkdir -p "$T"
tail -n +2 splits/train.csv | cut -d, -f2 > "$T/lista.txt"

$PY kaggle/sam_paineis.py --root "$DATA" --list "$T/lista.txt" --model "$SAM" --out "$T/sam"
$PY scripts/sam_objetos.py --root "$DATA" --list "$T/lista.txt" --model "$SAM" --out "$T/obj"
$PY scripts/treina.py --data "$DATA" --sam "$T/sam" --obj "$T/obj" --work "$T/cands" --workers "$WORKERS" --out tabelas
