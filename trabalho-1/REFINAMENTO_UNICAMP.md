# Refinamento visual da Unicamp

O resultado anterior apresentava mudanças de exposição e mistura de contornos
nos corrimãos. `refinar_unicamp.py` reproduz a correção da composição a partir
dos **42 JPGs originais** e da geometria congelada em
`configs/unicamp_refinamento.json`. Não modifica os padrões de `panorama.py`.

```bash
python trabalho-1/refinar_unicamp.py
python -m unittest discover -s trabalho-1/tests -v
```

O script valida os SHA-256 das 42 fotos antes de usar a calibração. A geometria
base foi estimada pelo bundle adjustment da branch, no commit registrado no
JSON: projeção cilíndrica, referência P1011561, escala de trabalho
0,30880082346886256. O JSON torna a renderização reproduzível sem depender de
caches pickle locais. Ele é específico deste conjunto, não uma calibração geral.

A composição usa compensação de exposição por blocos/canal, graph cut e mistura
multibanda. **A correção é assistida:** há duas prioridades manuais de fonte nos
corrimãos e três correspondências manuais que ajustam a fonte P1011599 por uma
transformação afim, restrita a um polígono de primeiro plano. Essas coordenadas
estão no JSON. O ajuste mantém linhas retas; não usa fluxo óptico, preenchimento
generativo ou conteúdo inventado. Não se deve apresentar esta etapa como
inteiramente automática nem atribuí-la ao algoritmo original da branch.

A imagem final mede **3828 × 924**, com o mesmo recorte/enquadramento anterior.
Cada fonte é renderizada diretamente do JPG original por um mapa inverso que
combina homografia e projeção cilíndrica. Isso evita ampliar o panorama pronto
ou encadear duas interpolações de imagens reduzidas. As costuras são estimadas
na resolução de trabalho e transferidas para a renderização final.

As etapas 1–4 continuam documentando a reconstrução global anterior ao ajuste
manual de primeiro plano. As oito saídas tradicionais da etapa 5 são
regeneradas; os métodos comparados recebem a mesma compensação de exposição e
o ajuste de primeiro plano. As métricas são calculadas em **1276 × 308**, sobre
fontes compensadas, e não são diretamente comparáveis às métricas antigas.
A linha `Graph cut assistido` identifica a composição escolhida; não significa
que todos os demais métodos também recebam prioridades manuais de costura.
`contribuicao_fontes.csv` informa os pixels selecionados de cada foto: uma fonte
registrada pode ser totalmente ocluída pela costura de outra. Todas as 42 são
verificadas, alinhadas e avaliadas.

A verificação combina cobertura integral do recorte, testes do mapa inverso e
das máscaras, leitura dos arquivos e inspeção visual do panorama e dos recortes.
Os testes não comprovam qualidade estética. Persistem pequenos erros de
paralaxe/degraus e distorção de perspectiva: homografias não modelam toda a
profundidade da cena nem o deslocamento da câmera.
