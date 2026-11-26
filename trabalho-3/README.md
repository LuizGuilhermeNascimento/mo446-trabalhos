# MO446 — Trabalho 3: detecção de cópia e colagem em figuras científicas

Dados e métrica da competição
[Recod.ai/LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection).
Para cada figura, a resposta é `authentic` ou uma máscara por região copiada.

## O que tem aqui

| Item pedido no enunciado | Onde está |
|---|---|
| Partições (treino, validação, teste) | `splits/train.csv`, `splits/val.csv`, `splits/test.csv` |
| Treino | `train.sh` |
| Predição | `predict.sh` |
| Predições da partição de teste | `predicoes_teste.csv` |
| Script que calcula S com a métrica oficial | `eval.py` (usa `metric/recodai_f1.py`, cópia sem modificações de [metric/recodai-f1](https://www.kaggle.com/code/metric/recodai-f1)) |
| Baseline clássico (SIFT) | `baseline_sift.py`, predições do teste em `baseline_teste.csv` |
| Caderno submetido ao Kaggle | `kaggle/submissao/run_submission.py` |

Demais pastas:

- `cmfd/` — o método (`cmfd/fluxo.py` junta as etapas, da figura à resposta);
- `kaggle/` — o que roda no Kaggle: fotos e texto pelo SAM (`sam_paineis.py`), resposta de cada imagem
  (`predict_fluxo.py`), caderno que baixa os pesos do SAM (`sam_pesos/`);
- `scripts/` — partições (`make_splits.py`), objetos pelo SAM (`sam_objetos.py`) e treino (`verif_cache.py`,
  `verif_treina.py`, `mascara_seletor.py`, `monta_tabelas.py`);
- `tabelas/` — as tabelas de treino usadas na submissão (saída do `train.sh`).

Os dados não estão aqui: baixe-os da competição. A pasta deve ter `train_images/`, `train_masks/`,
`supplemental_images/` e `supplemental_masks/`.

## Método

1. **Fotos, texto e objetos** — SAM 3.1 congelado (pesos públicos, sem ajuste) acha as fotos de cada figura
   (frase "microscopic images"), o texto e os objetos de dentro das fotos (células, manchas, bandas).
2. **Comparação entre fotos** — cada par de fotos da figura é comparado com SIFT e RANSAC; se um par casa, a resposta
   sai daqui.
3. **Procura dentro da figura** — SIFT (Kornia, na GPU) em 3 ampliações (lado maior 2048, 1536 e 1024), com CLAHE e
   imagem espelhada; vizinhos mais próximos exatos (g2NN); três famílias de candidatos (movimento afim por RANSAC,
   ligações confirmadas pelo ruído, pedaços fixos da figura).
4. **Máscara e medidas** — máscara por correlação local (ZNCC, janelas 7 e 21), refinada pelo ruído; 37 medidas por
   candidato.
5. **Verificador** — árvores com gradient boosting (400 iterações, taxa 0,05, 15 folhas) decidem se o candidato é
   cópia; fica quem tem probabilidade >= 0,9. Os aceitos são juntados em até 4 máscaras.
6. **Seletor de máscara** — duas regressões (árvores) estimam se trocar cada pedaço da máscara pelo objeto do SAM
   melhora; troca se o ganho estimado passa de 0,02.

Só os itens 5 e 6 têm pesos ajustados, e só com a partição de treino. Limiares e escolhas foram conferidos na
validação.

## Partições

Sorteio 70/15/15 com semente 42, estratificado pela classe. A imagem autêntica e a forjada de um mesmo caso ficam
sempre na mesma partição.

| Partição | Arquivo | Imagens | Sem fraude | Várias fotos | Únicas com fraude | Casos |
|---|---|---:|---:|---:|---:|---:|
| Treino | `splits/train.csv` | 3.624 | 1.664 | 34 | 1.926 | 1.957 |
| Validação | `splits/val.csv` | 777 | 357 | 7 | 413 | 420 |
| Teste | `splits/test.csv` | 775 | 356 | 7 | 412 | 418 |

Grupos: **Sem fraude** = autênticas (`train_images/authentic`); **Várias fotos** = figuras compostas com fraude
(`supplemental_images`); **Únicas com fraude** = as demais com fraude (`train_images/forged`).

Colunas: `case_id`, `path` e `mask_path` (relativos à raiz dos dados), `label` (`authentic` ou `forged`) e `source`
(`train` ou `supplemental`).

## Como rodar

Precisa de GPU (SAM 3.1 e SIFT do Kornia).

```bash
pip install -r requirements.txt
export DATA=<pasta dos dados da competição>

./train.sh                               # gera tabelas/candidatos.csv e tabelas/pedacos.csv
./predict.sh test predicoes_teste.csv    # predições da partição de teste
python eval.py --data $DATA --split test --submission predicoes_teste.csv   # S e nota por grupo
python eval.py --data $DATA --split test --baseline                         # baseline: tudo authentic
python baseline_sift.py --data $DATA --split val --varre 4,5,6,8,10,12,15,20,30,50,80        # baseline clássico: escolhe o mínimo de pares
python baseline_sift.py --data $DATA --split test --min-pares 8 --out baseline_teste.csv   # e roda o teste com ele
```

- `train.sh` roda, só na partição de treino: SAM (fotos, texto, objetos), candidatos com os cortes abertos,
  validação cruzada do verificador em 5 partes por caso e a tabela de pedaços do seletor. As saídas intermediárias
  ficam em `work/train/`.
- `predict.sh` roda os mesmos passos do caderno do Kaggle: SAM, depois ajusta o verificador e o seletor nas tabelas
  de `tabelas/` (alguns segundos) e responde cada imagem. Aceita `train`, `val` ou `test`.
- As tabelas da submissão já estão em `tabelas/`, então dá para rodar o `predict.sh` sem refazer o treino.
- Os pesos do SAM vêm de `unsloth/sam3.1` (Hugging Face); para usar uma pasta local, `export SAM=<pasta>`.
- Qualquer erro interrompe a execução: não existe resposta-padrão.

### Arquivo de predições

`predict.sh` grava `case_id`, `annotation` (formato de submissão: `authentic` ou as máscaras em RLE, pela função
oficial) e `path`. O `path` é a chave usada pelo `eval.py`, porque nas partições locais o mesmo `case_id` aparece
como imagem autêntica e como forjada. Imagem faltando, sobrando ou repetida é erro.

`eval.py` mostra S (média da nota oficial por imagem) no total e por grupo; com `--por-imagem notas.csv` grava a
nota de cada imagem.

## Resultados

Teste (775 imagens), rodado uma única vez, com tudo já fixado: `./predict.sh test predicoes_teste.csv` e `eval.py`.

| Grupo | Imagens | S | Marcadas como fraude |
|---|---:|---:|---:|
| Sem fraude | 356 | 0,9972 | 1 |
| Várias fotos | 7 | 0,3553 | 4 |
| Únicas com fraude | 412 | 0,6892 | 338 |
| **Todas** | 775 | **0,8276** | 343 |
| Baseline (tudo `authentic`) | 775 | 0,4594 | 0 |

Comparação com os baselines no teste (775 imagens: 356 sem fraude, 419 com fraude):

| Método | S | Acurácia | Precisão | Revocação | F1 | oF1 nas forjadas detectadas |
|---|---:|---:|---:|---:|---:|---:|
| Tudo `authentic` | 0,4594 | 0,4594 | – | 0,0000 | – | – |
| Baseline clássico (SIFT) | 0,6337 | 0,7239 | 0,9476 | 0,5179 | 0,6698 | 0,6778 |
| **Método proposto** | **0,8276** | 0,8994 | 0,9971 | 0,8162 | 0,8976 | 0,8375 |

Matriz de confusão no teste (linhas: verdade; colunas: resposta):

| | Método proposto: fraude | Método proposto: autêntica | Baseline SIFT: fraude | Baseline SIFT: autêntica |
|---|---:|---:|---:|---:|
| Com fraude (419) | 342 | 77 | 217 | 202 |
| Sem fraude (356) | 1 | 355 | 12 | 344 |

Acurácia, precisão, revocação e F1 são da decisão por imagem (fraude ou autêntica). A última coluna é a média da
nota oficial só nas imagens com fraude que o método detectou.

**Baseline clássico** (`baseline_sift.py`): pontos SIFT (Kornia, na GPU; lado maior limitado a 2048 px), cada ponto
casado com o mais parecido da própria imagem a mais de 20 px, teste de razão 0,6, um movimento (deslocamento, giro,
escala) por RANSAC e uma máscara pelo contorno convexo dos pontos concordantes dos dois lados. Responde fraude com
8 ou mais pares concordantes; esse mínimo foi escolhido na validação (S = 0,628 com 8; 0,580 com 4; 0,608 com 20) e
o teste rodou uma única vez com ele.

Validação (777 imagens), mesmos scripts:

| Grupo | Imagens | S | Marcadas como fraude |
|---|---:|---:|---:|
| Sem fraude | 357 | 0,9972 | 1 |
| Várias fotos | 7 | 0,5000 | 5 |
| Únicas com fraude | 413 | 0,7602 | 369 |
| **Todas** | 777 | **0,8667** | 375 |
| Baseline (tudo `authentic`) | 777 | 0,4595 | 0 |

Kaggle (late submission, mesmo código e mesmas tabelas): **0,35340** público e **0,46550** privado.

As duas rodadas locais usaram uma GPU NVIDIA B200, 16 processos e as versões do `requirements.txt` (scikit-learn
1.6.1, a mesma do Kaggle). Tempo: cerca de 19 minutos por partição (SAM de fotos e texto 0,08 s por imagem, SAM de
objetos 0,3 s, o restante 1,0 s por imagem de relógio).

## Caderno do Kaggle

`kaggle/submissao/run_submission.py` (internet desligada, 2 GPUs T4). Entradas:

- um Dataset com `cmfd/`, `kaggle/`, `scripts/`, `metric/` e `tabelas/` deste repositório;
- a saída do caderno `kaggle/sam_pesos/prepara.py`, que roda com internet e guarda os pesos do SAM 3.1 e os pacotes
  do `transformers`.

Ele roda o SAM (fotos e texto numa GPU, objetos na outra) em paralelo com a resposta das imagens e grava
`submission.csv` na ordem do `sample_submission.csv`.
