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
| Predições da partição de teste | `predicoes_teste.csv` (**ainda não gerado**: o teste roda uma única vez, no fim) |
| Script que calcula S com a métrica oficial | `eval.py` (usa `metric/recodai_f1.py`, cópia sem modificações de [metric/recodai-f1](https://www.kaggle.com/code/metric/recodai-f1)) |
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

Validação (777 imagens), pelo `predict_fluxo.py` deste repositório com `eval.py`:

| Grupo | Imagens | S |
|---|---:|---:|
| Sem fraude | 357 | 0,9888 |
| Várias fotos | 7 | 0,5000 |
| Únicas com fraude | 413 | 0,7620 |
| **Todas** | 777 | **0,8638** |
| Baseline (tudo `authentic`) | 777 | 0,4595 |

Essa medição usou scikit-learn 1.6.1 (a versão do Kaggle) e o SIFT do Kornia em CPU. Com scikit-learn 1.9.1 o total
é 0,8664: a versão muda um pouco as árvores.

Kaggle (late submission, mesmo código e mesmas tabelas): **0,35340** público e **0,46550** privado.

Teste local: ainda não rodado.

## Caderno do Kaggle

`kaggle/submissao/run_submission.py` (internet desligada, 2 GPUs T4). Entradas:

- um Dataset com `cmfd/`, `kaggle/`, `scripts/`, `metric/` e `tabelas/` deste repositório;
- a saída do caderno `kaggle/sam_pesos/prepara.py`, que roda com internet e guarda os pesos do SAM 3.1 e os pacotes
  do `transformers`.

Ele roda o SAM (fotos e texto numa GPU, objetos na outra) em paralelo com a resposta das imagens e grava
`submission.csv` na ordem do `sample_submission.csv`.
