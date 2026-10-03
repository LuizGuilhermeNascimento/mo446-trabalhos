# Panorama 360° — coleta de Bragança Paulista

As 22 fotos de `input/panorama360/` foram capturadas com Samsung Galaxy S24 Ultra em Bragança Paulista (SP). O local foi confirmado pelo autor; o dispositivo está no EXIF. A coleta é diurna e contém carros em movimento, fios e estruturas próximas. Os metadados de coleta estão em `configs/panorama360_coleta.json`.

## Executar

A partir da raiz do repositório, com as dependências de `trabalho-1/requirements.txt`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python trabalho-1/panorama_360.py \
  --input trabalho-1/input/panorama360 --output trabalho-1/output/panorama360/entrega
python trabalho-1/diagnosticos_360.py \
  --input trabalho-1/input/panorama360 --output trabalho-1/output/panorama360/entrega
python -m unittest discover -s trabalho-1/tests -v
```

O módulo é separado do pipeline anterior. Não muda seus parâmetros padrão. A execução foi validada com OpenCV 4.14.0. O HTML gerado contém o JPEG e funciona localmente sem servidor ou conexão externa; abra `visualizador_360.html`, arraste e use a roda para ajustar o campo de visão.

## Método

- Entrada embaralhada com semente 0; a geometria usa pixels, sem horários, GPS ou sequência dos nomes. `imread` aplica a orientação. Registro em maior lado de 1200 px; SIFT com até 5000 pontos.
- Correspondências OpenCV `BestOf2NearestMatcher(match_conf=0.3)` em todos os pares, com homografias robustas. Aceitação inicial: confiança > 1 e ao menos 25 inliers. Isso é uma configuração própria do experimento 360, diferente do ratio 0,75 dos três experimentos originais.
- Estimativa inicial por homografias, seguida de RANSAC sobre raios calibrados (500 amostras de três pares, limiar de 4 px multiplicando a distância entre raios unitários pela focal inicial). Esse filtro evita que correspondências incompatíveis com rotação contaminem o fechamento.
- Ajuste global robusto de 22 rotações e uma focal compartilhada (450–1100 px), referência fixa, `soft_l1` com escala 2, até 120 correspondências por par. Uma primeira passagem permite rejeitar pares com mediana residual > 15 px; uma segunda exige convergência. Os status de ambas são salvos.
- Projeção esférica equiretangular com todas as longitudes, recortando somente a maior faixa vertical continuamente coberta. A composição remapeia diretamente os JPGs originais. Bordas refletidas são usadas no filtro de mistura; máscaras impedem criar áreas sem observação.
- Exposição por blocos e canal, blocos 64 px na resolução de costura, duas passagens. Costuras por programação dinâmica (`COLOR_GRAD`) em três períodos; seleção do período central. Mistura multibanda com oito níveis e 512 px de extensão periódica.
- Sem pontos manuais, regiões de prioridade ou preenchimento generativo. O programa recusa saída com lacunas na faixa 360.

O registro inicial usa os componentes `detail` do OpenCV, como no [exemplo oficial de stitching detalhado](https://github.com/opencv/opencv/blob/4.x/samples/python/stitching_detailed.py). O ajuste robusto e a composição periódica estão neste repositório.

## Resultado medido

Saída de 6000 × 1106 px: 360° na horizontal e 66,36° na vertical; **não** é uma esfera completa de 360° × 180°. Todas as 22 imagens entraram no registro; 21 fornecem pixels depois da seleção de costuras, pois a vista redundante é dispensável para a composição.

O RANSAC de rotação descartou três pares candidatos. O ajuste convergiu com 40 pares, focal 824,605 px e mediana de 2,414 px no resíduo angular multiplicado pela focal (4248 correspondências amostradas). Esse valor não é a mesma métrica de reprojeção das tabelas dos três experimentos anteriores. O par que une início e fim tem 265 inliers e mediana de 0,953 px. Todos os pixels da faixa final possuem suporte; os hashes das 22 fontes foram conferidos.

`verificacao.json`, `render.json`, `cameras_ajustadas.json`, as correspondências e os CSVs documentam esses números. `fechamento.jpg` mostra as últimas e primeiras colunas lado a lado. `mascara.png` confirma o suporte, mas não mede fidelidade geométrica. Permanecem paralaxe na calçada e em objetos próximos, pequenas descontinuidades nos fios e efeitos de carros em movimento.

## Organização no Drive

A pasta original foi movida para `input/panorama360`. As saídas seguem a organização existente: `output/step1_key_points_detection/panorama360` até `output/step5_panorama_composition/panorama360`, e `output/final/panorama360`. Os arquivos de trabalho e tentativas intermediárias ficam apenas localmente.
