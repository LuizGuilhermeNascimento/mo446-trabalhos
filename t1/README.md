# T1 — Panorama automático

Pipeline completo em `panorama.ipynb`: leitura da pasta (nomes e EXIF ignorados), SIFT/ORB,
ratio test, RANSAC, grafo de conectividade com rejeição de intrusa, ajuste global de focal e
rotações, projeção cilíndrica, compensação de exposição, costura por programação dinâmica e
blending multibanda.

## Executar

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name mo446
.venv/bin/python executar.py --entrada <pasta_de_imagens> --saida outputs
```

`executar.py` roda o notebook num kernel novo e grava `outputs/panorama.png`, figuras,
CSVs e `metricas.json`. As imagens (DNG/JPG/PNG) não estão no repositório.

## Verificar

```bash
.venv/bin/python tools/validar_execucao.py       # 5 embaralhamentos de nomes (requer uma execução prévia)
```

As funções vivem apenas no notebook, nas células marcadas `definicoes`; a validação carrega
essas células, sem implementação paralela.
