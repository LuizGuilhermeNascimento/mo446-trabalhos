# T2 — Reconstrução 3D e comparação de objetos (Variante B)

O grupo fará a **Variante B**, pois o último dígito da soma dos RAs dos
integrantes é ímpar. O trabalho comparará **vários objetos** e, para cada
objeto, poderá incluir vídeos com diferentes condições de captura, como
exposição, abertura e ISO. O primeiro objeto é `pedra_cubica`; ele é a captura
inicial do conjunto, não o único objeto da análise. Para cada captura serão
registrados os metadados e os resultados de SIFT, ORB e AKAZE. O objetivo é
reconstruir os objetos e comparar como o objeto e as condições de captura
afetam a reconstrução 3D.

## Organização dos dados

Cada vídeo e sua sequência de quadros devem ser identificados por objeto e
captura. O vídeo inicial existente permanece em `images/pedra_cubica/`; novas
capturas devem usar uma subpasta própria para não misturar quadros:

```text
images/
├── pedra_cubica/
│   ├── P1011579.MOV                 # vídeo inicial
│   ├── frame_0001_t00000.00s.jpg    # quadros da captura inicial
│   └── captura_iso1600/             # exemplo de outra exposição
│       ├── video.mov
│       └── frame_0001_t00000.00s.jpg
└── outro_objeto/
    └── captura_base/
        ├── video.mov
        └── frame_0001_t00000.00s.jpg
```

Cada entrada da lista `DATASETS` no início de `reconstruction.ipynb` representa
uma captura de um objeto. Para adicionar uma nova condição de exposição, crie
uma entrada com o mesmo `object_name`, um `capture_id` diferente, o caminho do
vídeo, uma pasta exclusiva para os quadros e os parâmetros efetivamente usados
na gravação. Para adicionar outro objeto, use um novo `object_name`. A seleção
`ACTIVE_DATASET_IDS` determina quais capturas são processadas e comparadas na
execução atual.

O notebook gera `informacoes_captura_<captura>_<N>quadros.txt` em cada pasta
de quadros e imprime seu
conteúdo ao carregar a captura. Assim os metadados permanecem vinculados às
imagens e também são apresentados durante a execução. Não se deve copiar os
parâmetros de exposição de uma gravação para outra: registre os valores reais
de cada vídeo.

As visualizações, tabelas e cópias dos metadados são salvas em
`outputs/<object_name>/`, espelhando o nome da pasta em `images/`. Para a
captura inicial, os arquivos ficam em `outputs/pedra_cubica/`; os nomes das
figuras incluem o identificador da captura e o método.

## Captura inicial: pedra_cubica

- Vídeo: `images/pedra_cubica/P1011579.MOV`.
- Câmera: Olympus OM-D E-M5 Mark II.
- A folha de papel quadriculado com espaçamento de 5 mm está visível em todos os quadros extraídos, segundo a inspeção da captura, e fornece a referência de escala. A calibração de `K` pelo papel é experimental: ondulações e interseções cobertas podem aumentar o erro, por isso registre o RMS e interprete a calibração como aproximada.
- Configurações informadas para esta gravação: foco manual (MF), abertura
  f/5.6, tempo de exposição 1/125 s e ISO 800.
- A folha de papel quadriculado com espaçamento de 5 mm está visível em todos os quadros extraídos. A seleção manual da grade é feita somente no notebook `preparacao_calibracao.ipynb`; a pipeline `reconstruction.ipynb` carrega a calibração salva e não pede cliques. Os parâmetros e as sobreposições ficam em `outputs/<object_name>/<capture_id>_<N>quadros_calibracao/` e são reutilizados enquanto os quadros e as opções permanecerem iguais.
- A tarefa recomenda de 40 a 120 imagens, com 60% a 80% de sobreposição entre
  vistas vizinhas, cobrindo o objeto em duas ou três alturas. O plano é extrair
  80 quadros uniformemente distribuídos por vídeo.
- A captura deve favorecer luz difusa e evitar superfícies reflexivas ou
  translúcidas. É necessário incluir em várias imagens uma referência de escala
  conhecida e medir pelo menos três dimensões reais do objeto.

Para extrair os quadros, a função recebe explicitamente o caminho do vídeo:

```bash
python extrair_imagens.py --video images/pedra_cubica/P1011579.MOV --frames 80 --saida images/pedra_cubica
```

O resultado da primeira captura é salvo em `images/pedra_cubica/`. Para outra
captura, informe o vídeo e uma pasta de saída exclusivos. O notebook chama a
função com os caminhos indicados em `DATASETS`. O runner `executar.py` também
chama a função; para reutilizar as imagens existentes sem extrair de novo,
execute `python executar.py --usar-imagens-existentes`.

## Ambiente Python

Com Python 3.12, instale as dependências de execução com:

```bash
python -m pip install -r requirements.txt
```

As dependências de desenvolvimento, como pytest, estão em
`requirements-dev.txt`. COLMAP, caso seja usado como executável, deve ser
instalado separadamente do ambiente Python.

## Instalação do COLMAP para a Etapa 4

COLMAP é um programa externo; ele não é instalado por `requirements.txt`.
Instale uma versão com suporte CUDA para executar PatchMatch MVS e as etapas
densas desta pipeline. A instalação do COLMAP deve ser feita uma vez em cada
computador que executará a Etapa 4. Consulte os binários e as instruções oficiais
em [COLMAP — Installation](https://colmap.github.io/install.html).

### Windows

1. Baixe o pacote binário de Windows com CUDA e extraia-o, por exemplo, em
   `C:\Apps\COLMAP`.
2. Confirme que `COLMAP.bat` está nessa pasta e, no PowerShell aberto nessa
   pasta, execute `.\COLMAP.bat -h`.
3. Na célula de configuração da Etapa 4 em `reconstruction.ipynb`, informe o
   caminho completo:

   ```python
   COLMAP_EXECUTABLE = r"C:\Apps\COLMAP\COLMAP.bat"
   ```

   Se o diretório do COLMAP já estiver no `PATH`, deixe `COLMAP_EXECUTABLE = None`.

### Linux

1. Instale uma versão CUDA do COLMAP. O pacote do gerenciador de pacotes da
   distribuição pode ser instalado com `sudo apt install colmap`, mas os pacotes
   dos repositórios padrão normalmente não incluem suporte CUDA. Para a Etapa 4,
   use um build CUDA, compile o COLMAP com CUDA ou execute o contêiner CUDA
   oficial.
2. No terminal, confirme que `colmap -h` funciona. Se `colmap` estiver no
   `PATH`, deixe `COLMAP_EXECUTABLE = None` no notebook. Caso contrário, informe
   o caminho para o executável, por exemplo:

   ```python
   COLMAP_EXECUTABLE = "/usr/local/bin/colmap"
   ```

Em qualquer sistema, o computador precisa de uma GPU NVIDIA e de um driver
compatível com o build CUDA utilizado para executar PatchMatch MVS. `pip freeze`
ou `requirements.txt` não registra o executável COLMAP, o driver da GPU nem o
CUDA Toolkit do sistema.

## Pipeline

1. **Extração e comparação de características:** para cada captura, visualizar
   keypoints e comparar **SIFT, ORB e AKAZE** nas mesmas imagens. Comparar
   também resultados entre objetos e entre capturas do mesmo objeto feitas com
   exposições diferentes. Registrar metadados, tempo de execução, quantidade de
   keypoints, matches, inliers e erro geométrico; justificar o método usado nas
   etapas seguintes.
2. **Emparelhamento e geometria epipolar:** usar BFMatcher com a distância
   apropriada ao descritor, aplicar o ratio test de Lowe e, quando necessário,
   a verificação cruzada mútua. Estimar `F` com RANSAC e comparar SIFT, ORB e
   AKAZE por pares: quantidade e taxa de inliers, cobertura espacial e erro de
   Sampson. Salvar CSV e gráfico no estilo da análise de inliers do T1, além de
   visualizar os três métodos no mesmo par de frames. Com a calibração `K`,
   estimar `E`, recuperar `R` e `t` e triangular uma nuvem 3D inicial. Exibir
   matches inliers e linhas epipolares; documentar que `t` e os pontos ainda
   têm escala monocular relativa.
3. **SfM e escala:** gerar a nuvem esparsa e as poses das câmeras; converter o
   modelo para escala métrica usando a referência de tamanho conhecido.
4. **Densificação e superfície:** executar MVS, gerar nuvem densa, reconstruir
   malha (por exemplo, Poisson) e produzir a textura. Visualizar e limpar
   outliers quando necessário.
5. **Avaliação:** informar erro médio e mediano de reprojeção, contagem de
   pontos esparsos e densos, densidade aproximada em pontos/m² e erros absoluto
   e relativo das dimensões medidas (erro dimensional em mm). Comparar uma foto
   real com a renderização do modelo no mesmo ponto de vista e discutir
   costuras, borrões e regiões sem cobertura.
6. **Ablação e comparação de capturas:** comparar diferentes objetos,
   exposições, quantidades de imagens e alturas de captura; registrar seus
   efeitos nos descritores, erro métrico e completude da reconstrução. Manter
   constantes os demais parâmetros sempre que possível e identificar cada
   conjunto de resultados com seu `object_name` e `capture_id`.

## Tarefas adicionais

O grupo pretende realizar também as tarefas opcionais indicadas na folha:

- Marcar as poses das câmeras no espaço 3D e exportar o modelo para GLB ou OBJ
  para visualização web.
- Analisar casos difíceis da captura, como textura insuficiente, iluminação,
  oclusões ou superfícies brilhantes.
- Comparar PatchMatch com um método alternativo de densificação.
- Se a malha resultar fechada, estimar o volume do objeto e compará-lo com uma
  medição real por deslocamento de água.

## Reprodutibilidade e entregáveis

O repositório final deve fornecer `run.sh` para executar o pipeline e um script
de visualização. A entrega inclui relatório técnico em PDF de até 6 páginas,
apresentação oral de 20 a 25 minutos e demonstração com vídeo de navegação de
30 a 60 segundos. Para a Variante B, são obrigatórias a nuvem esparsa, a nuvem
densa e a malha texturizada. O prazo informado é 4 de novembro de 2026.

## Estado do código

`extrair_imagens.py` recebe explicitamente o caminho de cada vídeo. A captura
inicial possui 80 quadros em `images/pedra_cubica/`; `NUM_FRAMES` permite
alterar a amostragem. Os diretórios e arquivos de resultados por captura
incluem o identificador e a quantidade de quadros (por exemplo,
`pedra_cubica_base_80quadros_etapa4`) para comparar execuções. `reconstruction.ipynb`
mantém os metadados e a seleção de múltiplas capturas.

## Preparação antes da pipeline

Execute `preparacao_calibracao.ipynb` antes de `reconstruction.ipynb` para cada
captura nova ou quando mudar o vídeo, a quantidade de quadros, os índices de
calibração ou o tamanho conhecido da quadrícula. O notebook de preparação
reutiliza os quadros existentes quando a quantidade solicitada já está
disponível; caso contrário, extrai os quadros do vídeo configurado. Em seguida,
abre uma janela para cada um dos sete quadros finais e solicita quatro cliques
por imagem para estimar `K` e a escala do grid. Os pontos selecionados são
marcados por cruzes finas. O cache e os diagnósticos são gravados em
`outputs/<objeto>/<captura>_<N>quadros_calibracao/`.

Depois de salvar a calibração, execute `reconstruction.ipynb` normalmente. A
célula de calibração da pipeline apenas valida os nomes dos quadros e carrega o
cache; ela falha com uma instrução clara se a preparação estiver ausente ou
desatualizada. Assim, o processamento principal não depende de interação
manual. Para outra captura, atualize os parâmetros em ambos os notebooks e rode
a preparação antes da pipeline. A seleção dos pontos abre uma janela independente
do Tkinter, então não depende de cliques em figuras exibidas pelo notebook. Se a
janela não aparecer ao executar a célula no PyCharm, execute a preparação no
terminal integrado, a partir da pasta `t2`, com
`.venv\Scripts\python.exe preparar_calibracao.py`. A janela pede os quatro
cantos dos sete quadros finais um após o outro.

A Etapa 4 está implementada em `etapa4_colmap.py`: SfM esparso, PatchMatch
MVS, duas fusões densas (consistência geométrica e fotométrica), malha Poisson,
textura, escala métrica baseada no papel com quadrícula de **5 mm** e previews.
Ela gera artefatos de reconstrução; não calcula métricas de qualidade nem erros
contra dimensões ou volume de referência. Para executar, instale as dependências
de `requirements.txt` e configure o executável COLMAP conforme a seção
"Instalação do COLMAP para a Etapa 4"; defina `COLMAP_EXECUTABLE` no notebook se
ele não estiver no `PATH`. A preparação
interativa da calibração deve ser feita antes, conforme descrito acima. A superfície do papel
deve ser suficientemente plana; o script interrompe a escala se as arestas
reconstruídas forem insuficientes ou inconsistentes.

Antes da execução, preencha em cada entrada de `DATASETS`:

```python
"camera": {
    # demais metadados da câmera...
    "expected_dimensions_mm": {"maior": None, "intermediaria": None, "menor": None},
    "expected_volume_cm3": None,
}
```

Insira as dimensões reais medidas em milímetros, em ordem decrescente (`maior`,
`intermediaria`, `menor`), e o volume de referência em cm³ (por exemplo, medido
por deslocamento de água). A Etapa 5 está implementada em
`etapa5_avaliacao.py` e grava `relatorio_etapa5_<N>quadros.json`, com erros
absoluto e relativo por dimensão e pelo volume, densidade aproximada em pontos/m²,
erro médio e mediano de reprojeção e contagens de pontos esparsos e densos. O
volume reconstruído só é informado se a malha for fechada (`watertight`). A
comparação lado a lado entre foto e render é criada na Etapa 4 e referenciada
no relatório da Etapa 5. Execute primeiro a célula da Etapa 4 e depois a célula
independente da Etapa 5 no notebook.

Os artefatos exportados têm nomes em português e incluem a quantidade real de
quadros da execução, por exemplo `nuvem_densa_geometrica_metrica_80quadros.ply`,
`malha_texturizada_metrica_80quadros.obj` e
`foto_e_render_lado_a_lado_80quadros.png`. Os arquivos internos que o COLMAP
exige mantêm os nomes esperados pelo próprio programa.
