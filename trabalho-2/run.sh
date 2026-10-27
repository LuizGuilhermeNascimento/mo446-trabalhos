#!/usr/bin/env bash
# Uso: ./run.sh dados/<objeto> [saida]    RES=2 reduz a resolução do MVS; RAPIDO=1 para após a avaliação.
set -euo pipefail
DADOS=$(cd "${1:?uso: ./run.sh dados/<objeto> [saida]}" && pwd)
SAIDA=${2:-$(dirname "$0")/saida/$(basename "$DADOS")}
mkdir -p "$SAIDA" && SAIDA=$(cd "$SAIDA" && pwd)
cd "$(dirname "$0")"
MVS=${OPENMVS_BIN:-$HOME/.local/openmvs/bin/OpenMVS}
PY=.venv/bin/python
W=$SAIDA/mvs

$PY sfm.py "$DADOS" "$SAIDA"
$PY escala.py "$DADOS" "$SAIDA"
"$MVS/InterfaceCOLMAP" -w "$W" -i . -o scene.mvs --image-folder images
"$MVS/DensifyPointCloud" -w "$W" -i scene.mvs -o scene_dense.mvs --resolution-level "${RES:-1}"
"$MVS/ReconstructMesh" -w "$W" -i scene_dense.mvs -p scene_dense.ply -o scene_mesh.ply
$PY avaliar.py "$DADOS" "$SAIDA"
[ -n "${RAPIDO:-}" ] && exit 0
"$MVS/RefineMesh" -w "$W" -i scene_dense.mvs -m scene_mesh.ply -o scene_refine.ply --resolution-level 1
"$MVS/TextureMesh" -w "$W" -i scene_dense.mvs -m scene_refine.ply -o scene_texture.mvs --export-type obj
$PY features.py "$DADOS" "$SAIDA"
$PY visualizar.py "$DADOS" "$SAIDA"
