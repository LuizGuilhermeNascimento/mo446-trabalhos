# MO446 — Trabalho 3: partições e avaliação

Detecção de cópia e colagem (copy-move) em figuras científicas, com os dados da competição
[Recod.ai/LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection).

Este repositório guarda só o necessário para avaliar um método de forma reproduzível:

- `splits/` — as três partições (treino, validação e teste) das imagens rotuladas da competição;
- `eval.py` — nota oficial de um arquivo de respostas numa partição, no total e por grupo;
- `metric/recodai_f1.py` — a métrica oficial, copiada sem modificações do caderno
  [metric/recodai-f1](https://www.kaggle.com/code/metric/recodai-f1);
- `scripts/make_splits.py` — recria as partições a partir dos dados.

Os dados não estão aqui: baixe-os da competição. A pasta deve ter `train_images/`, `train_masks/`,
`supplemental_images/` e `supplemental_masks/`.

## Partições

Sorteio 70/15/15 com semente 42. A imagem autêntica e a forjada de um mesmo caso ficam sempre na mesma partição.

| Partição | Arquivo | Imagens | Sem fraude | Várias fotos | Únicas com fraude | Casos |
|---|---|---:|---:|---:|---:|---:|
| Treino | `splits/train.csv` | 3.624 | 1.664 | 34 | 1.926 | 1.957 |
| Validação | `splits/val.csv` | 777 | 357 | 7 | 413 | 420 |
| Teste | `splits/test.csv` | 775 | 356 | 7 | 412 | 418 |

Os grupos:

- **Sem fraude**: imagens autênticas (`train_images/authentic`);
- **Várias fotos**: figuras compostas com fraude (`supplemental_images`);
- **Únicas com fraude**: as demais imagens com fraude (`train_images/forged`).

Colunas dos arquivos: `case_id`, `path` e `mask_path` (relativos à raiz dos dados), `label` (`authentic` ou `forged`)
e `source` (`train` ou `supplemental`).

Uso das partições: o método é ajustado no treino, as escolhas são conferidas na validação, e o teste roda uma única
vez, no fim.

## Avaliação

```bash
pip install -r requirements.txt
python eval.py --data <raiz dos dados> --split val --submission submission.csv
```

O arquivo de respostas tem uma linha por imagem da partição e duas colunas:

| Coluna | Conteúdo |
|---|---|
| `path` | caminho da imagem, como em `splits/<partição>.csv` |
| `annotation` | `authentic` ou as máscaras no formato da competição (`rle_encode` de `metric/recodai_f1.py`) |

A chave é o caminho, e não o `case_id`, porque o mesmo `case_id` aparece como imagem autêntica e como forjada.
Imagem faltando, sobrando ou repetida é erro.

A saída mostra a nota (média da nota oficial por imagem) em cada grupo e no total:

```
Partição: val (777 imagens) | respostas: tudo authentic

Grupo                 Imagens     Nota   Marcadas como fraude
Sem fraude                357   1.0000                      0
Várias fotos                7   0.0000                      0
Únicas com fraude         413   0.0000                      0
Todas                     777   0.4595                      0
```

Esse exemplo é a resposta `authentic` em todas as imagens (`python eval.py --data <raiz> --split val --baseline`).
Com `--por-imagem notas.csv` o script grava também a nota de cada imagem.

Para montar a coluna `annotation` a partir de máscaras binárias (uma por região copiada):

```python
import sys; sys.path.insert(0, "metric")
import recodai_f1
annotation = recodai_f1.rle_encode(masks) if masks else "authentic"   # masks: lista de arrays (H, W) com 0 e 1
```
