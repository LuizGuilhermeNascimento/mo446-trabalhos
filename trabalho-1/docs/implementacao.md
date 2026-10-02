# Implementação do Pipeline de Panorama: Decisões e Resultados

Documento de apoio ao relatório (E1) e à apresentação (E4). Explica o que cada etapa implementa, as escolhas feitas e os resultados obtidos no conjunto **paisagem**. Todos os números vêm de `python panorama.py input/paisagem`, e as figuras citadas ficam em `output/<passo>/paisagem/`.

| Etapa do enunciado | Pasta do código | Arquivo principal |
|---|---|---|
| 2. Detecção e extração de características | `step1_key_points_detection/` | `detector.py` |
| 3. Emparelhamento de características | `step2_feature_matching/` | `matcher.py` |
| 4. Ordenação automática | `step3_image_sorting/` | `sorter.py` |
| 5. Homografia e alinhamento | `step4_homography_connection/` | `homography.py`, `alignment.py` |
| 6. Composição, blending e fantasmas | `step5_panorama_composition/` | `compositor.py`, `evaluation.py` |

**Visão geral do conjunto paisagem:**
- 9 fotos da Pedra Grande (Atibaia) tiradas com iPhone 15, de pé, girando a câmera da esquerda para a direita, mais 1 intrusa de outra cena (`20260926_7`). Os nomes de arquivo não seguem a ordem de captura: a intrusa recebeu o número 7.
- As imagens são reduzidas para 1024 px no lado maior. Cada uma cobre ~69° de campo de visão, e o panorama completo, ~175°.
- O pipeline completo leva ~40 s: detecção 12 s, emparelhamento 8 s, ordenação 2 s, alinhamento 5 s, composição 12 s.

---

## Etapa 1. Coleta e entrada

Os dados da coleta (item 1.4) ficam em `input/paisagem/metadados.json` e são exibidos no início da execução. Para atender ao item 4.1, a lista de arquivos é **embaralhada** com uma semente fixa antes de qualquer processamento, e o algoritmo usa apenas os pixels. Os nomes servem só de rótulo, e nenhum metadado EXIF de tempo é lido.

Ordem recebida pelo pipeline nesta execução: `7, 8, 10, 5, 3, 4, 2, 1, 9, 6`.

---

## Etapa 2. Detecção e Extração de Características

### Detectores comparados

| Detector | Como detecta | Descritor | Características |
|---|---|---|---|
| **SIFT** | Extremos da Diferença de Gaussianas (DoG) numa pirâmide de escalas; orientação pelo histograma de gradientes ao redor do ponto | 128 valores reais (histogramas de gradiente em uma grade 4×4), comparados pela distância L2 | Invariante a escala e rotação, robusto a iluminação. O mais custoso |
| **ORB** | Cantos FAST numa pirâmide de imagens, ordenados pelo score de Harris; orientação pelo centroide de intensidade | BRIEF rotacionado: 256 bits, comparados pela distância de Hamming | Muito rápido; menos robusto a mudanças grandes de escala |
| **AKAZE** | Espaço de escala **não linear** (difusão anisotrópica), que suaviza regiões homogêneas e preserva bordas | M-LDB binário: 488 bits, distância de Hamming | Invariante a escala e rotação; custo intermediário |

Os três usam o mesmo limite de **3000 keypoints por imagem**, mantendo os de maior resposta, para a comparação ser justa.

### Keypoints com escala e orientação

Cada keypoint tem uma **posição**, uma **escala** e uma **orientação**:
- A **escala** é o tamanho da vizinhança usada para descrever o ponto. Um mesmo canto pode ser detectado como pequeno numa imagem próxima e grande numa imagem afastada, e o descritor é calculado nessa escala. É isso que dá invariância a escala.
- A **orientação** é a direção dominante do gradiente local. O descritor é calculado num referencial girado para essa direção, o que dá invariância a rotação.

Na figura `detectores_<imagem>.png`, cada keypoint aparece como um ponto. Nos 300 de maior resposta, o **raio do círculo é a escala** e o **segmento é a orientação**, como na Figura 2 do enunciado.

### Comparação qualitativa (mesma imagem)

Na figura `detectores_20260926_5.png`:
- **SIFT:** espalha os keypoints pela cidade, pela vegetação e pelas bordas das nuvens, com círculos pequenos, em muitas escalas finas.
- **ORB:** concentra os pontos em poucos aglomerados, sobretudo na cidade densa, e os círculos são grandes. O ORB usa uma vizinhança fixa de 31 px, ampliada pelos níveis da pirâmide.
- **AKAZE:** fica entre os dois, com menos pontos, mas bem localizados nas bordas fortes.

**Regiões homogêneas são um limite da cena.** O céu ocupa cerca de 40% de cada foto e quase não gera keypoints em nenhum detector. Os poucos pontos do céu ficam nas nuvens e no parapente. Isso aparece na **cobertura da grade 8 × 8**: nenhum detector passa de ~65%, e as células vazias são as do céu. Como discutido em aula, cenas com grandes áreas homogêneas não são as mais recomendadas para panoramas baseados em características. Aqui elas funcionaram porque a faixa inferior (cidade e mata) tem textura de sobra, mas o céu fica sem "âncoras" para o alinhamento.

### Comparação quantitativa (média ± desvio padrão sobre as 10 imagens)

| Detector | Keypoints | Tempo (ms) | Cobertura 8×8 | Uniformidade |
|---|---|---|---|---|
| SIFT | 3000 ± 0 | 44.4 ± 2.5 | 65% ± 7 | 0.82 ± 0.02 |
| ORB | 2979 ± 42 | 14.2 ± 0.5 | 51% ± 12 | 0.65 ± 0.06 |
| AKAZE | 1244 ± 186 | 33.3 ± 0.6 | 62% ± 6 | 0.76 ± 0.05 |

- **Cobertura:** porcentagem das 64 células da grade 8×8 que têm ao menos um keypoint.
- **Uniformidade:** entropia normalizada da contagem por célula, onde 1 significa distribuição perfeitamente homogênea.
- SIFT e ORB atingem o limite de 3000 keypoints. O AKAZE não tem parâmetro de quantidade e encontra menos pontos.

### Trade-off e escolha: SIFT

- **O ORB é ~3× mais rápido** (14 ms contra 44 ms por imagem), mas concentra os pontos (cobertura de 51%, uniformidade de 0.65) e tem a menor precisão de emparelhamento (87%, ver Etapa 3).
- **O SIFT é o mais lento,** mas tem a melhor cobertura e uniformidade e a maior precisão de emparelhamento (95.9%). Pontos bem espalhados estabilizam a homografia, porque um modelo ajustado só a uma região da imagem extrapola mal para o resto.
- **Escala do estudo:** com 10 imagens, a diferença de tempo entre SIFT e ORB é de ~0.3 s num pipeline de ~40 s. O que se busca aqui é **qualidade e robustez do alinhamento, não tempo real**, e por isso o SIFT foi escolhido. Para vídeo ou centenas de imagens, o ORB seria a escolha natural.

---

## Etapa 3. Emparelhamento de Características

### Brute Force em vez de FLANN

- **Brute Force (BF):** compara cada descritor de uma imagem com **todos** os da outra e devolve os *k* vizinhos mais próximos **exatos**. O custo é O(N·M) por par: com 3000 × 3000 descritores, ~9 milhões de distâncias. Usa distância **L2** para o SIFT e **Hamming** para os descritores binários (ORB e AKAZE).
- **FLANN:** busca **aproximada** de vizinhos com índices: KD-trees aleatórias para descritores reais e LSH para binários. Compensa com dezenas de milhares de descritores ou mais, mas pode errar o vizinho mais próximo e depende de parâmetros (número de árvores, *checks*, tabelas LSH).

**Por que BF:**
1. **É exato.** O ratio test depende dos dois vizinhos mais próximos *reais*, e um vizinho aproximado errado muda a decisão do teste.
2. **Nesta escala, é mais rápido.** Num estudo preliminar com 3000 descritores, o FLANN levou ~44 ms por par contra ~13 ms do BF, porque o índice precisa ser reconstruído a cada par. A ordenação emparelha **todos** os pares (45 neste conjunto).
3. **É determinístico e sem parâmetros extras.** A mesma entrada sempre dá o mesmo resultado.

### Ratio test de Lowe e o limiar 0.75

Para cada descritor, o BF retorna os **dois** vizinhos mais próximos, *d₁* ≤ *d₂*. O match é **aceito** se:

> *d₁* < 0.75 · *d₂*

A ideia é aceitar apenas matches **sem ambiguidade**. Se o segundo candidato é quase tão parecido quanto o primeiro (textura repetitiva, céu, folhagem), o match é descartado, mesmo que o primeiro esteja certo.

**Por que 0.75 e não o 0.8 sugerido por Lowe (2004):** o limiar foi escolhido pelos dados, com a **curva do ratio test** (`curva_ratio_test.png`). Ela mede, para vários limiares, quantos matches são aceitos e que fração deles é geometricamente consistente (precisão):

| Limiar (SIFT) | 0.70 | **0.75** | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|
| Matches aceitos por par | 677 | **700** | 727 | 784 | 955 |
| Precisão | 97.1% | **95.9%** | 93.3% | 84.9% | 68.9% |

Passar de 0.75 para 0.8 ganha só 4% de matches e perde 2.6 pontos de precisão. Na cena noturna da igreja a diferença é maior: 74.4% contra 65.2%. **A intenção foi priorizar matches corretos:** com ~700 matches por par, ainda há muito mais do que o necessário para estimar uma homografia (4 pontos), e menos outliers deixam o RANSAC e a ordenação mais confiáveis.

### Métricas e comparação entre detectores (BF, limiar 0.75)

As médias abaixo são sobre os **28 pares com sobreposição**. A coluna "Falsos aceitos" é a média sobre os 17 pares sem sobreposição.

| Detector | Tempo/par | Aceitos | Taxa de aceitação | Inliers | **Precisão** | Falsos aceitos |
|---|---|---|---|---|---|---|
| SIFT | 14.1 ms | 700 | 23.3% | 679 | **95.9%** | 13.5 |
| ORB | 4.6 ms | 542 | 18.1% | 505 | 87.4% | 5.0 |
| AKAZE | 2.8 ms | 343 | 26.7% | 332 | 94.1% | 5.5 |

- **Taxa de aceitação:** aceitos pelo ratio test ÷ candidatos.
- **Precisão:** inliers ÷ aceitos. Como não há gabarito de quais matches estão corretos, usa-se um **RANSAC apenas como régua de avaliação**: um match é considerado correto se for consistente com a homografia do par (erro menor que 3 px).
- **Falsos aceitos:** matches aceitos em pares que **não** se sobrepõem, que são necessariamente errados.

O SIFT tem a maior precisão e o maior número de inliers, o que confirma a escolha da Etapa 2. Ele aceita mais matches falsos em pares sem sobreposição (13.5 por par), mas esses matches não formam uma geometria consistente e o RANSAC os descarta (Etapa 4).

### Desafios observados (requisito 3.4)

As figuras `matches_<a>_<b>.jpg` mostram a mesma amostra de candidatos antes do ratio test (amarelo) e depois dele (verde = aceito, vermelho tracejado = descartado):
- **Galhos próximos da câmera** (paralaxe): os galhos no topo das primeiras imagens estão a poucos metros, enquanto a paisagem está a quilômetros. Quando a câmera gira, eles se deslocam de forma diferente do fundo e ainda balançam com o vento. Geram matches longos e cruzados, quase todos descartados pelo ratio test.
- **Parapente (objeto móvel):** a mesma vela aparece em posições diferentes em cada foto. O descritor é idêntico, então o ratio test **aceita** o match, mas ele é geometricamente incorreto (não segue o movimento da câmera) e precisa ser removido pela verificação geométrica.
- **Céu e textura repetitiva:** o céu quase não gera candidatos, e a mata gera candidatos ambíguos, descartados pelo ratio test.
- **Par sem sobreposição** (`matches_sem_sobreposicao_*.jpg`, a intrusa contra uma imagem da cena): mesmo assim alguns matches passam no ratio test, o que mostra que o ratio test sozinho não basta e é preciso uma verificação geométrica.

---

## Etapa 4. Ordenação Automática das Imagens (obrigatória)

### Passo a passo

1. **Entrada:** 10 imagens embaralhadas, 9 da cena e 1 intrusa, sem qualquer informação de ordem (Etapa 1).
2. **Emparelhamento de todos os pares:** para cada um dos 45 pares, SIFT + BF + ratio test e, em seguida, uma **homografia com RANSAC**. O número de **inliers** de cada par forma a **matriz de conectividade** (`matriz_conectividade.png/.csv`).
3. **Verificação de plausibilidade:** uma homografia degenerada (espelhada, ou que leva a região dos pontos para trás da câmera) é descartada, e o par fica com 0 inliers.
4. **Decisão automática de sobreposição**, pelo critério de Brown & Lowe (2007): o par se sobrepõe se
   > inliers ≥ max(20, 8 + 0.3 · matches)

   O limiar cresce com o número de matches: um par com muitos matches, mas poucos geometricamente consistentes, é rejeitado. Isso evita aceitar pares que só compartilham textura parecida (mata, céu). O resultado é a **matriz de adjacência** (`matriz_adjacencia.csv`).
5. **Grafo de vizinhança:** cada imagem é um nó, e cada par que se sobrepõe é uma aresta com peso igual ao número de inliers. No conjunto paisagem há **28 arestas**: as fotos se sobrepõem até 4 vizinhas de distância.
6. **Rejeição de intrusas:** o **maior componente conexo** do grafo é o panorama, e qualquer imagem fora dele é intrusa. Aqui, `20260926_7` teve **0 inliers** com a melhor candidata (`20260926_8`), contra um mínimo de 20, e ficou isolada (`intrusa_20260926_7.jpg`, `grafo_vizinhanca.png`).

### Da matriz à sequência

7. **Árvore geradora máxima:** do grafo do panorama, mantém-se a árvore que liga todas as imagens com a **maior soma de inliers**. Cada imagem fica ligada ao vizinho mais confiável, e as conexões "de longa distância" (1–5, 2–6…) são descartadas.
8. **Sequência = caminho mais longo da árvore** (diâmetro): uma busca em largura a partir de qualquer nó encontra o extremo *u*, e uma segunda a partir de *u* encontra o outro extremo *v*. O caminho de *u* até *v* é a ordem do panorama. Se a árvore tiver ramificações, as imagens fora do caminho vão para o final, com um aviso.
9. **Sentido:** projeta-se o centro da segunda imagem no referencial da primeira pela homografia. Se ele cair à esquerda, a sequência é invertida, o que garante a ordem da esquerda para a direita.
10. **Referência:** a imagem **central** da sequência (posição 5), que minimiza a distorção acumulada nas pontas.

**Resultado:** a sequência inferida foi `1 → 2 → 3 → 4 → 5 → 6 → 8 → 9 → 10`, exatamente a ordem de captura. Os pares vizinhos têm entre 755 e 1315 inliers, contra limiares de 238 a 407, uma margem ampla. Figuras: `sequencia_inferida.jpg`, `grafo_vizinhanca.png`.

---

## Etapa 5. Estimação de Homografia e Alinhamento

### Homografia e RANSAC

Uma **homografia** *H* (matriz 3×3, 8 graus de liberdade) mapeia pontos de uma imagem para outra: *x'* ~ *H x*. Ela é exata quando a câmera **apenas gira** em torno do centro óptico (ou quando a cena é plana), que é o caso de um panorama feito de pé no mesmo ponto.

**RANSAC** (`cv2.findHomography`, limiar de 3 px, até 2000 iterações, confiança de 99.5%):
1. sorteia 4 correspondências e calcula a homografia exata que elas definem;
2. conta quantas das demais correspondências caem a menos de 3 px da posição prevista (inliers);
3. repete e mantém o modelo com mais inliers, reajustado com todos os seus inliers.

É robusto a outliers como o parapente, os galhos e os matches ambíguos, porque esses pontos não concordam com o modelo da maioria.

### Métricas dos pares utilizados (requisito 5.3)

Os pares utilizados são os vizinhos da sequência, usados para encadear as homografias (`metricas_homografias.png`):

| Par | 1–2 | 2–3 | 3–4 | 4–5 | 5–6 | 6–7 | 7–8 | 8–9 |
|---|---|---|---|---|---|---|---|---|
| Taxa de inliers | 97.9% | 95.4% | 98.8% | 95.9% | 99.8% | 99.6% | 98.6% | 92.6% |
| Erro de reprojeção médio | 0.96 px | 0.89 px | 0.59 px | 0.96 px | 0.71 px | 0.57 px | 0.60 px | 0.53 px |

Todos os pares têm taxa de inliers acima de 92% e erro abaixo de 1 px, um alinhamento sub-pixel. Os pares das pontas (1–2 e 8–9) têm as menores taxas, porque contêm os galhos e a rocha em primeiro plano, com paralaxe.

### Projeção cilíndrica, e não planar

A projeção **planar** projeta todas as imagens no plano da imagem de referência. Um raio a um ângulo θ do eixo óptico cai a *f*·tan θ do centro, o que **diverge quando θ se aproxima de 90°**. O conjunto paisagem cobre ~175°, então as imagens das pontas ficariam esticadas ao infinito. Medido neste conjunto, a projeção planar geraria um canvas de **19094 × 10536 px**, e o pipeline recusa a execução nesse caso.

A projeção **cilíndrica** mapeia cada pixel para um cilindro de raio *f*: *x'* = *f*·arctan(*x*/*f*), *y'* = *f*·*y*/√(*x*² + *f*²). Um giro horizontal da câmera vira uma **translação** no cilindro, e o panorama cresce linearmente com o ângulo. O canvas fica com **2391 × 795 px**.

**A distância focal *f* é estimada automaticamente:** para uma câmera que só gira, *H* = *K R K⁻¹*. O pipeline busca o *f* (em *K*) que torna *K⁻¹ H K* mais próxima de uma rotação para todas as arestas da árvore. Resultado: **748 px**, o que corresponde a ~69° de campo de visão por imagem, coerente com a câmera grande-angular do iPhone. Imagens e pontos são então projetados no cilindro, e as homografias são reestimadas.

### Alinhamento global com `warpPerspective`

- **Encadeamento par a par:** a partir da referência, cada imagem recebe a transformação *G* = *G*ₚₐᵢ · *H*(imagem → pai), percorrendo a árvore em largura.
- **Warp:** uma translação comum leva todo o panorama para coordenadas positivas, e cada imagem (e sua máscara de validade) é transformada para o canvas com `cv2.warpPerspective` (requisito 5.2).
- As máscaras perdem 2 px de borda, porque a interpolação mistura a borda da imagem com o preto de fora.

Figuras: `alinhamento_progressivo.jpg` (o canvas a cada nova imagem, com a recém-adicionada contornada em amarelo; requisito 5.4) e `contornos_imagens.jpg`.

### Bundle adjustment (extra X1)

O encadeamento par a par usa **apenas a árvore** (8 pares) e acumula pequenos erros ao longo da cadeia (deriva). O **bundle adjustment** refina **todas as transformações ao mesmo tempo**, usando **todas as 28 sobreposições**, inclusive as que fecham ciclos (1–3, 2–5…):
- **Parâmetros:** os 8 parâmetros de *G* de cada imagem; a referência fica fixa na identidade.
- **Resíduos:** erro de transferência **simétrico** dos inliers de cada par (*i* → *j* e *j* → *i*), com até 200 pontos por par.
- **Otimização:** `scipy.optimize.least_squares` com perda de **Huber** (escala de 2 px), robusta a outliers remanescentes, a partir da solução par a par. Converge em ~0.5 s.

| Erro de reprojeção médio | Pares vizinhos (8) | Todas as sobreposições (28) | Erro máximo |
|---|---|---|---|
| Par a par | **0.73 px** | 1.03 ± 0.72 px | 4.27 px |
| Bundle adjustment | 0.80 px | **0.80 ± 0.27 px** | **1.53 px** |

**Interpretação:** o par a par é ótimo nos pares vizinhos *por construção*, porque usa exatamente a homografia de cada um. O bundle adjustment distribui o erro: aceita um pequeno aumento nos vizinhos (+0.07 px) e, em troca, reduz o erro médio global em 22% e o **erro máximo em quase 3×**, ou seja, elimina a deriva.

**Qual escolher:** o bundle adjustment melhora a **consistência geométrica global**, mas **não melhora o panorama final** nestes conjuntos. As métricas da composição ficam praticamente iguais nos dois modos, e os panoramas são visualmente indistinguíveis:

| Conjunto | Modo | Erro fotométrico entre vizinhas | Pixels discordantes | Fantasmas (final) | Descontinuidade na costura |
|---|---|---|---|---|---|
| paisagem | par a par | 6.61 | 3.30% | 0.09% | 0.56 |
| paisagem | bundle | 6.75 | 3.43% | 0.07% | 0.53 |
| igreja | par a par | 17.64 | 17.83% | 0.17% | 0.47 |
| igreja | bundle | 17.80 | 18.00% | 0.16% | 0.43 |

A composição costura cada imagem principalmente às suas **vizinhas**, e nesses pares o par a par já é ótimo. A deriva que o BA corrige aparece nos pares distantes, que quase não influenciam o resultado. Por isso o pipeline mantém o **par a par** (`alignment = "pairwise"`): é mais simples, é o método pedido pelo enunciado e dá o mesmo panorama. O BA fica como extra, com o ganho medido no erro global. Ele faria diferença em panoramas que fecham um ciclo (360°) ou com cadeias mais longas, em que a deriva acumulada vira desalinhamento visível.

---

## Etapa 6. Composição, Blending e Remoção de Fantasmas (obrigatória)

### Métodos comparados

| Método | Remoção de fantasmas | Blending | Como funciona |
|---|---|---|---|
| Média simples | não | não | Cada pixel é a média das imagens que o cobrem (composição ingênua, requisito 6.1) |
| Feathering | não | sim | Média ponderada; o peso de cada imagem cresce com a distância à sua borda (`distanceTransform`) |
| Costura | **sim** | não | Cada pixel vem de **uma única** imagem, separada das outras por costuras ótimas |
| **Costura + feathering** | **sim** | **sim** | Costura ótima, com suavização gaussiana (σ = 15 px) **apenas em uma faixa estreita em torno da costura** |
| Costura + multibanda | sim | sim | Costura ótima e blending de Burt & Adelson: pirâmides laplacianas (5 níveis), com frequências baixas misturadas numa faixa larga e detalhes só junto à costura |

**Costura ótima por programação dinâmica** (requisito 6.3, Figura 6 do enunciado): as imagens entram no mosaico a partir da referência. Para cada imagem nova:
1. o custo é a **diferença de cor** entre o mosaico atual e a nova imagem na sobreposição, levemente suavizada para a costura não passar rente a bordas de objetos;
2. a programação dinâmica encontra o **caminho de menor custo acumulado** que atravessa a sobreposição, como no *seam carving*;
3. de um lado da costura os pixels vêm do mosaico, do outro, da nova imagem.

A costura passa onde as imagens **concordam**, e por isso **contorna objetos em movimento**: o parapente fica inteiro numa só imagem. O mapa `costuras.jpg` mostra a imagem de origem de cada pixel.

O resultado final é recortado no **maior retângulo sem bordas pretas**: de 2391 × 795 px para **2180 × 620 px**.

### Remoção de fantasmas (requisito 6.4)

Um **fantasma** surge quando imagens com conteúdo diferente no mesmo lugar são **misturadas**. Em `comparacao_deghosting.jpg`, a região do parapente mostra:
- **média simples:** **6 cópias translúcidas** do parapente, uma por foto em que ele aparece;
- **costura + feathering:** um **único parapente nítido**, vindo de uma só foto.

O método que remove fantasmas é a **costura ótima**: como cada pixel vem de uma só imagem, nada é misturado. Feathering e multibanda, sozinhos, só suavizam a transição e **não removem** fantasmas: o feathering sem costura também mistura as cópias.

### Métricas (requisito 6.5)

`metricas_metodos_composicao.png/.csv`:

| Método | Pixels com fantasma | Descontinuidade na costura | Tempo |
|---|---|---|---|
| Média simples | 2.90% | 0.53 | 0.6 s |
| Feathering | 2.42% | 0.53 | 0.8 s |
| Costura | **0.00%** | 2.29 | 1.4 s |
| **Costura + feathering** | **0.09%** | **0.56** | 1.5 s |
| Costura + multibanda | 0.03% | 1.05 | 1.9 s |

- **Pixels com fantasma:** pixels da sobreposição onde as imagens alinhadas **discordam** (mais de 30 níveis em algum canal de cor) e o panorama **não coincide com nenhuma delas**, ou seja, mostra uma mistura. A comparação é feita em cor porque o parapente rosa sobre o céu azul tem quase o mesmo nível de cinza.
- **Descontinuidade na costura:** salto de intensidade entre pixels vizinhos separados por uma costura, **descontado** o salto que já existe na própria imagem de origem. O valor 0 significa uma costura invisível. Todos os métodos são medidos nas mesmas costuras.
- **Erro fotométrico na sobreposição** (`erro_fotometrico_sobreposicao.csv`): diferença média entre imagens vizinhas alinhadas, de 4.4 a 10.1 níveis de cinza. Os maiores valores (pares 1–2 e 2–3) vêm dos galhos com paralaxe.
- **Distorção geométrica** (`distorcao_por_imagem.csv`): desvio máximo dos ângulos dos cantos de cada imagem em relação a 90°, no máximo **2.1°**. A projeção cilíndrica praticamente não distorce as imagens.

### Melhor método, trade-offs e dificuldades

**Costura + feathering é o melhor equilíbrio:** é o único método com fantasmas **e** descontinuidade baixos ao mesmo tempo.
- **Costura sem blending:** zero fantasmas, mas o corte é ~4× mais visível (2.29), sobretudo onde a costura cruza o **galho** próximo da câmera (`comparacao_metodos.jpg`).
- **Média e feathering:** transições suaves, mas fantasmas em ~2.5–3% dos pixels sobrepostos.
- **Multibanda:** também remove fantasmas, mas cria um leve **halo** perto da costura (descontinuidade de 1.05). As frequências baixas se misturam numa faixa larga, e as imagens diferem em exposição e no conteúdo próximo da câmera.

**Trade-off central:** quanto mais se mistura, mais suave fica a transição e mais fantasmas aparecem. O feathering **estreito** (σ = 15 px) só mistura junto à costura, onde a costura ótima já escolheu um lugar em que as imagens concordam, e por isso não reintroduz os fantasmas.

**Dificuldades:**
- **Paralaxe dos galhos:** nenhuma homografia alinha ao mesmo tempo o primeiro plano e o fundo. A costura evita parte do problema, mas onde ela precisa cruzar um galho surge um corte. É uma limitação física da captura, não do algoritmo.
- **Objeto móvel:** a costura pode manter o parapente uma vez (o caso aqui), mas, em geral, poderia mantê-lo duas vezes ou cortá-lo, conforme o caminho da costura.
- **Exposição automática:** pequenas variações de brilho entre as fotos tornam as costuras levemente perceptíveis no céu. Uma compensação de ganho (extra X3) não foi implementada.
- **Céu homogêneo:** sem keypoints, o alinhamento do céu depende de a homografia estimada na faixa inferior se estender corretamente para cima.
