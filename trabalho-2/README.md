# T2 — Reconstrução 3D com múltiplas imagens (Variante B: objeto)

Fotos → SfM (pycolmap, CPU) → escala métrica pelo tabuleiro ChArUco → MVS denso, malha e textura (OpenMVS, CPU) → métricas e visualizações.

## Instalação (Ubuntu 26.04)

```bash
sudo apt install -y cmake libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev libglew-dev libglfw3-dev \
  libeigen3-dev libboost-iostreams-dev libboost-program-options-dev libboost-serialization-dev \
  libboost-exception-dev libopencv-dev libcgal-dev libgmp-dev libmpfr-dev zlib1g-dev
./instalar_openmvs.sh   # OpenMVS 2.4.0 sem CUDA em ~/.local/openmvs
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
```

No macOS (Apple Silicon), troque o `apt` por:

```bash
brew install cmake pkgconf boost cgal eigen libomp libusb uv
JOBS=8 ./instalar_openmvs.sh   # também compila uma OpenCV 4.12 mínima, porque a do Homebrew é 5.x
```

`instalar_openmvs.sh` aplica quatro correções de uma linha à v2.4.0: Boost ≥ 1.89, JPEG XL opcional, OpenCV 4.10 e as texturas pretas da v2.4.0 (issue #1257, corrigida no upstream).

## Dados

O protocolo de coleta está em [COLETA.md](COLETA.md). Imprima `charuco_a4.pdf`, `charuco_a3.pdf` ou `charuco_a2.pdf` (gerados por `charuco.py`) em 100%. A estrutura esperada é:

```
dados/<objeto>/{h1,h2,h3,topo}/*.jpg
dados/<objeto>/medidas.json
```

## Execução

```bash
./run.sh dados/<objeto>                       # saída em saida/<objeto>/
.venv/bin/python ablacao.py dados/<objeto>    # saída em saida/<objeto>_ablacao/
```

`RES=2 ./run.sh ...` reduz a resolução do MVS; `RAPIDO=1` para depois da avaliação, sem refino, textura nem figuras.

| Etapa | Script | Saída em `saida/<objeto>/` |
|---|---|---|
| 2–3 | `features.py` | `features.json`, `figuras/{keypoints,inliers,epipolares,triangulacao}_*` |
| 4 SfM | `sfm.py` | `esparso/` |
| 4 escala | `escala.py` | `metrico/`, `roi/`, `mvs/` (imagens sem distorção), `escala.json` |
| 4 MVS | OpenMVS | `mvs/scene_dense.ply`, `scene_mesh.ply`, `scene_refine.ply`, `scene_texture.obj` |
| 5 | `avaliar.py` | `metricas.json`, `objeto.ply` |
| 5 | `visualizar.py` | `figuras/cameras.png`, `figuras/foto_vs_render_*`, `cena.glb`, `navegacao.mp4` |
| 5 | `ablacao.py` | `ablacao.json`, `ablacao.png` |

## Método

- **Escala:**
  - Os cantos do ChArUco são detectados em todas as fotos registradas.
  - Cada canto é triangulado por DLT com as poses do SfM.
  - Uma similaridade (Umeyama) leva o modelo ao referencial do tabuleiro, em mm, com a mesa em z = 0. Ela é aplicada antes do MVS, então todos os produtos já saem métricos.
- **Medidas:**
  - O objeto é a maior componente da malha acima de z = 2 mm.
  - Comprimento e largura são os lados do menor retângulo que contém a vista de cima; a altura é o z máximo.
  - O volume vem do teorema da divergência com F = (0, 0, z): a base apoiada na mesa contribui com zero.
- **Ablação:**
  - Subconjuntos por número de alturas e de fotos por anel.
  - Erro métrico, mais acurácia e completude (τ = 1 e 2 mm) em relação à reconstrução com todas as fotos.
