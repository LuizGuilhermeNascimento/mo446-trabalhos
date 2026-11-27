# MO446 — Trabalho 3: cópia e colagem em figuras científicas

Competição [Recod.ai/LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection).

| Item do enunciado | Arquivo |
|---|---|
| Partições | `splits/train.csv`, `splits/val.csv`, `splits/test.csv` |
| Treino | `train.sh` |
| Predição | `predict.sh` |
| Predições do teste | `predicoes_teste.csv` |
| Script do S (métrica oficial, sem modificações, em `metric/`) | `eval.py` |
| Baseline clássico (SIFT) | `baseline_sift.py`, `baseline_teste.csv` |
| Caderno do Kaggle | `kaggle/submissao/run_submission.py` |

O método está em `cmfd/` (`cmfd/fluxo.py` junta as etapas); `scripts/` tem as partições e o treino; `tabelas/` é a
saída do treino usada na submissão.

## Como rodar

Precisa de GPU. Os dados vêm da competição (`train_images/`, `train_masks/`, `supplemental_images/`, `supplemental_masks/`).

```bash
pip install -r requirements.txt
export DATA=<pasta dos dados>
./train.sh                                                                  # gera tabelas/
./predict.sh test predicoes_teste.csv
python eval.py --data $DATA --split test --submission predicoes_teste.csv   # S, total e por grupo
python eval.py --data $DATA --split test --baseline                         # tudo authentic
python baseline_sift.py --data $DATA --split val --varre 4,5,6,8,10,12,15,20,30,50,80
python baseline_sift.py --data $DATA --split test --min-pares 8 --out baseline_teste.csv
```

`tabelas/` já vem preenchida, então o `predict.sh` roda sem refazer o treino. Os arquivos de predição têm `case_id`,
`annotation` e `path`; o `path` é a chave, porque o mesmo `case_id` aparece como autêntica e como forjada.

## Método

1. SAM 3.1 congelado acha as fotos, o texto e os objetos de cada figura.
2. Comparação entre fotos (SIFT + RANSAC); se um par casa, a resposta sai daqui.
3. Senão, SIFT na GPU em 3 ampliações (2048, 1536, 1024), com CLAHE e espelho, gera candidatos de cópia.
4. Máscara de cada candidato por correlação local (ZNCC) e 37 medidas.
5. Verificador (árvores com gradient boosting) aceita os candidatos com probabilidade >= 0,9.
6. Seletor de máscara (duas regressões) decide se troca cada pedaço pelo objeto do SAM.

Só os itens 5 e 6 são treinados, e só com a partição de treino.

## Partições

70/15/15, semente 42, agrupado por `case_id` e estratificado pela composição de cada caso (só forjada, ou autêntica e
forjada) e pela origem (principal ou suplementar).

| Partição | Imagens | Sem fraude | Várias fotos | Únicas com fraude |
|---|---:|---:|---:|---:|
| Treino | 3.624 | 1.664 | 34 | 1.926 |
| Validação | 777 | 357 | 7 | 413 |
| Teste | 775 | 356 | 7 | 412 |

## Resultados no teste

| Método | S | Acurácia | Precisão | Revocação | F1 | oF1 nas forjadas detectadas |
|---|---:|---:|---:|---:|---:|---:|
| Tudo `authentic` | 0,4594 | 0,4594 | – | 0,0000 | – | – |
| Baseline clássico (SIFT) | 0,6337 | 0,7239 | 0,9476 | 0,5179 | 0,6698 | 0,6778 |
| **Método proposto** | **0,8276** | 0,8994 | 0,9971 | 0,8162 | 0,8976 | 0,8375 |

| S por grupo | Sem fraude | Várias fotos | Únicas com fraude | Todas |
|---|---:|---:|---:|---:|
| Teste | 0,9972 | 0,3553 | 0,6892 | 0,8276 |
| Validação | 0,9972 | 0,5000 | 0,7602 | 0,8667 |

| Matriz de confusão (teste) | Proposto: fraude | Proposto: autêntica | SIFT: fraude | SIFT: autêntica |
|---|---:|---:|---:|---:|
| Com fraude (419) | 342 | 77 | 217 | 202 |
| Sem fraude (356) | 1 | 355 | 12 | 344 |

Kaggle (mesmo código e mesmas tabelas): 0,35340 público e 0,46550 privado, no
[notebook de submissão](https://www.kaggle.com/code/thiagolaitz/mo446-t3-submissao-fluxo).

Baseline clássico: SIFT casado com a própria imagem, teste de razão 0,6, RANSAC e máscara pelo contorno dos pontos
concordantes; responde fraude com 8 ou mais pares (mínimo escolhido na validação).

Rodadas locais numa NVIDIA B200 com 16 processos: cerca de 19 minutos por partição.
