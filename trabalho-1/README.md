# Trabalho 1: Panoramas

Pipeline que recebe uma pasta de imagens **fora de ordem** (com uma imagem intrusa) e produz o panorama final sem intervenção manual (entregável E3 de `docs/T1.pdf`).

## Estrutura

```
trabalho-1/
├── panorama.py                   # orquestrador: executa os passos 1-5 (parâmetros no Config)
├── common.py                     # leitura das imagens, pastas de saída, CSV, tabelas e estilo das figuras
├── requirements.txt
├── input/<grupo>/                # fotos de um grupo (.jpg, .jpeg ou .png) + metadados.json da coleta
├── output/
│   ├── stepN_<nome>/<grupo>/     # figuras e CSVs de cada passo (não versionados)
│   └── final/<grupo>/panorama.jpg
├── step1_key_points_detection/   # SIFT x ORB x AKAZE
├── step2_feature_matching/       # BF x FLANN + ratio test de Lowe
├── step3_image_sorting/          # matriz de conectividade, grafo, ordem e intrusa
├── step4_homography_connection/  # RANSAC, projeção cilíndrica, par a par x bundle adjustment
└── step5_panorama_composition/   # costura ótima, feathering/multibanda, fantasmas, recorte
```

## Instalação

Requer Python 3.10 ou superior. A partir de `trabalho-1/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Execução

```bash
.venv/bin/python panorama.py input/landscape                # todos os passos de um grupo
.venv/bin/python panorama.py input/landscape --ate-passo 3  # para depois do passo N (1 a 5)
.venv/bin/python panorama.py --todos                        # todos os grupos de input/
```

Cada grupo leva cerca de 1 minuto. O terminal mostra:
- os parâmetros usados;
- a ordem de entrada (embaralhada);
- as tabelas de cada passo;
- o resumo final: sequência inferida, intrusas rejeitadas, referência e tamanho do panorama;
- o tempo de cada passo.

## Adicionar um grupo próprio

1. Crie `input/<nome_do_grupo>/` e copie as fotos para ela. Os nomes e a ordem dos arquivos não importam: o pipeline embaralha a entrada e nunca usa nomes nem EXIF.
2. Use **pelo menos 6 fotos** da mesma cena, com sobreposição entre fotos vizinhas (~30–50%), girando a câmera sem transladar. De preferência, inclua um elemento móvel (pessoa, veículo…) e **uma imagem intrusa** de outra cena.
3. Os formatos aceitos são `.jpg`, `.jpeg` e `.png`. Converta HEIC ou DNG antes, por exemplo no macOS com `sips -s format png foto.dng --out foto.png`.
4. Rode `.venv/bin/python panorama.py input/<nome_do_grupo>`.

## Configuração

Os métodos de cada passo ficam no `Config`, no topo de `panorama.py`. Cada campo tem um comentário explicando a escolha.

| campo | padrão | opções |
|---|---|---|
| `detector` | `SIFT` | `SIFT`, `ORB`, `AKAZE` |
| `matcher` / `ratio` | `BF` / `0.75` | `BF`, `FLANN` |
| `projection` | `cylindrical` | `cylindrical`, `planar` (a planar só serve para até ~120° de campo de visão) |
| `alignment` | `pairwise` | `pairwise`, `bundle` (bundle adjustment) |
| `deghost` / `blend` | `seam` / `feather` | `seam`, `none` / `feather`, `multiband`, `none` |
| `max_side`, `max_features`, `shuffle_seed`, `ghost_crop` | 1024, 3000, 0, `None` | resolução, keypoints, semente do embaralhamento, região extra na figura de fantasmas |

## Saídas em `output/<passo>/<grupo>/`

Cada passo gera as saídas do método escolhido e o estudo comparativo, em figuras e CSVs. Os mesmos números aparecem em tabelas no terminal.

| passo | principais arquivos |
|---|---|
| 1. detecção | `detectores_<imagem>.png` (keypoints de cada detector; círculo = escala, segmento = orientação dos 300 mais fortes), `comparacao_detectores.png/.csv`, `comparacao_detectores_por_imagem.csv` |
| 2. emparelhamento | `matches_<a>_<b>.jpg` (antes/depois do ratio test), `comparacao_matchers.png/.csv`, `curva_ratio_test.png/.csv`, `matches_sem_sobreposicao_*.jpg` |
| 3. ordenação | `matriz_conectividade.png/.csv`, `matriz_adjacencia.csv`, `grafo_vizinhanca.png`, `sequencia_inferida.jpg`, `intrusa_<imagem>.jpg`, `decisao_por_imagem.csv` |
| 4. homografia | `estatisticas_homografias.csv` (taxa de inliers e erro de reprojeção), `ransac_<a>_<b>.jpg`, `alinhamento_progressivo.jpg`, `contornos_imagens.jpg`, `comparacao_alinhamento.png` |
| 5. composição | `comparacao_deghosting.jpg` (mesmas regiões sem/com deghosting), `costuras.jpg`, `comparacao_metodos.jpg/.png/.csv`, `metricas_composicao.csv`, `distorcao_por_imagem.csv` |

O panorama final fica em `output/final/<grupo>/panorama.jpg`, recortado no maior retângulo sem bordas pretas.
