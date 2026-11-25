#!/usr/bin/env bash
# Predições do fluxo final para uma partição (padrão: teste), no formato de submissão.
#
# Uso:  DATA=<pasta dos dados da competição> ./predict.sh [particao] [saida]
#       particao = train, val ou test (padrão test); saida padrão = predicoes_teste.csv
# Opcionais: SAM (pasta ou nome dos pesos do SAM 3.1; padrão unsloth/sam3.1), WORK (padrão work), WORKERS (padrão 4),
#            PYTHON (padrão python).
# Precisa de GPU e das tabelas de treino (tabelas/, geradas pelo train.sh). São os mesmos passos do caderno do Kaggle
# (kaggle/submissao/run_submission.py). Saída: csv com case_id, annotation e path (o mesmo case_id aparece como
# imagem autêntica e como forjada; o path desfaz a ambiguidade). Qualquer erro interrompe o script.
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA:?defina DATA com a pasta dos dados da competição}"
PART=${1:-test}; SAIDA=${2:-predicoes_teste.csv}
SAM=${SAM:-unsloth/sam3.1}; WORK=${WORK:-work}; WORKERS=${WORKERS:-4}; PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f "splits/$PART.csv" ] || { echo "ERRO: não existe splits/$PART.csv"; exit 1; }
for f in tabelas/candidatos.csv tabelas/pedacos.csv; do [ -f "$f" ] || { echo "ERRO: falta $f (rode ./train.sh)"; exit 1; }; done
P="$WORK/$PART"; mkdir -p "$P"
tail -n +2 "splits/$PART.csv" | cut -d, -f2 > "$P/lista.txt"

# 1. SAM 3.1 congelado: fotos e texto de cada figura; objetos de dentro das fotos
$PY kaggle/sam_paineis.py --root "$DATA" --list "$P/lista.txt" --model "$SAM" --out "$P/sam" --batch 8 --dtype fp16 --texto
$PY scripts/sam_objetos.py --root "$DATA" --list "$P/lista.txt" --model "$SAM" --out "$P/obj"

# 2. ajusta verificador e seletor nas tabelas do treino e responde cada imagem
$PY kaggle/predict_fluxo.py --root "$DATA" --list "$P/lista.txt" --sam "$P/sam" --obj "$P/obj" \
    --tab-ver tabelas/candidatos.csv --tab-sel tabelas/pedacos.csv --sift kornia --workers "$WORKERS" --out "$P/respostas.csv"

# 3. formato de submissão
$PY - "$P/respostas.csv" "$SAIDA" <<'PYEOF'
import sys
import pandas as pd
R = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
assert (R.annotation.str.strip() != "").all(), "resposta vazia"
R[["case_id", "annotation", "path"]].to_csv(sys.argv[2], index=False)
print(f"{sys.argv[2]}: {len(R)} imagens, {int((R.annotation != 'authentic').sum())} marcadas como fraude")
PYEOF
echo "PREDICAO_OK"
