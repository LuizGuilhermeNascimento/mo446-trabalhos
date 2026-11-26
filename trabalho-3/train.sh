#!/usr/bin/env bash
# Treino, só com a partição de treino. Uso: DATA=<dados da competição> ./train.sh
# Opcionais: SAM, WORK, WORKERS, PYTHON. Precisa de GPU. Saída: tabelas/candidatos.csv e tabelas/pedacos.csv.
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA:?defina DATA com a pasta dos dados da competição (train_images/, train_masks/, supplemental_*/)}"
SAM=${SAM:-unsloth/sam3.1}; WORK=${WORK:-work}; WORKERS=${WORKERS:-4}; PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
T="$WORK/train"; mkdir -p "$T"
tail -n +2 splits/train.csv | cut -d, -f2 > "$T/lista.txt"

# 1. SAM 3.1 congelado: fotos e texto de cada figura; objetos de dentro das fotos
$PY kaggle/sam_paineis.py --root "$DATA" --list "$T/lista.txt" --model "$SAM" --out "$T/sam" --batch 8 --dtype fp16 --texto
$PY scripts/sam_objetos.py --root "$DATA" --list "$T/lista.txt" --model "$SAM" --out "$T/obj"

# 2. candidatos de cópia de cada figura (cortes abertos), com as medidas e o acerto de cada um
$PY scripts/verif_cache.py --data "$DATA" --sam "$T/sam" --sam-b "$T/sam" --split splits/train.csv --sift kornia --workers "$WORKERS" --out "$T/cache"

# 3. verificador: validação cruzada em 5 partes por caso; grava a tabela de candidatos com rótulo e probabilidade
$PY scripts/verif_treina.py --cache "$T/cache" --sam-b "$T/sam" --cortes 0.9 --workers "$WORKERS" --out "$T/verificador"

# 4. seletor de máscara: tabela de pedaços das máscaras aceitas com corte 0,9
$PY scripts/mascara_seletor.py --sem-regras --corte 0.9 --data "$DATA" --cache "$T/cache" --notas "$T/verificador/candidatos.csv" \
    --obj "$T/obj" --sam-b "$T/sam" --split splits/train.csv --workers "$WORKERS" --out "$T/seletor"

# 5. tabelas finais (só as colunas que os modelos usam)
$PY scripts/monta_tabelas.py --ver "$T/verificador/candidatos.csv" --sel "$T/seletor/pedacos.csv" --out tabelas
echo "TREINO_OK"
