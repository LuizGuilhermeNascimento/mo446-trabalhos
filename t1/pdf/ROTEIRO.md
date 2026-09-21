# Roteiro da apresentação

Duração prevista: 21.5 minutos. Abra `outputs/slides.html`; use setas para navegar, F para tela cheia e N para notas. O PDF oferece uma cópia para projeção ou envio.

Distribuam as seções entre os integrantes do grupo e ensaiem a passagem de uma etapa para a seguinte. Os nomes e a divisão final dependem do grupo.

## 1. Construção automática de panoramas (1 min)

Apresentar o objetivo: receber uma pasta sem ordem conhecida, reconhecer as vistas e montar o panorama. Mostrar o resultado para situar as seis etapas.

## 2. Sete vistas da rua e uma imagem externa (1.5 min)

Distinguir os sete DNG da imagem pública acrescentada. Explicar que os valores RAW de 16 bits não foram revelados. Mencionar iluminação noturna como limitação da coleta.

## 3. As seis etapas da implementação (1 min)

Explicar a dependência: o grafo usa inliers, então a estimação preliminar de homografias acontece antes da discussão geométrica completa. O código central fica visível no notebook.

## 4. ORB extrai mais rápido, SIFT tem menor resíduo (1.5 min)

Diferenciar detector e descritor. Apontar círculos e orientações dos keypoints. Tempos são medianas de três extrações por imagem, somadas; não são duração do pipeline. Erros usam inliers de conjuntos diferentes.

## 5. O segundo vizinho ajuda a medir ambiguidade (1.5 min)

Percorrer os três painéis do mesmo par: candidatos, ratio test e inliers. Explicar por que uma janela repetida pode ser visualmente parecida e geometricamente incorreta. O valor 0,80 foi avaliado nesta coleta.

## 6. Sete imagens formam um componente conectado (1.5 min)

Explicar os vértices e pesos. Foram avaliados os 28 pares. Mínimos de 25 inliers, taxa de 20%, cobertura de 0,4% e plausibilidade da transformação. I08 não tem aresta aceita.

## 7. A vista apontada para cima continua válida (1 min)

Responder à dúvida sobre a intrusa: mudança de enquadramento não implica outra cena. Explicar que a ordem vem da geometria, não de um percurso arbitrário da árvore. Mostrar a conexão fraca na extremidade.

## 8. RANSAC estima a transformação entre duas vistas (1.5 min)

Apresentar a convenção de H e a divisão homogênea. DLT é mostrado explicitamente; a estimação robusta é do OpenCV. Citar erro médio e taxa por aresta no relatório. Distinguir transformação de pixels e validade.

## 9. O cilindro acomoda a amplitude da varredura (2 min)

Explicar por que um único plano produz distorção excessiva nas extremidades. O ajuste estima focal comum e rotações, fixando uma câmera. Todas as arestas contribuem. O resíduo mostrado não é o erro 2D de reprojeção. O modelo não contém translação ou profundidade.

## 10. Ganhos nas sobreposições reduzem diferenças de cor (1 min)

A comparação mantém alinhamento e mistura por média. Estimar razões medianas por canal excluindo regiões muito escuras e saturadas. Os ganhos são limitados entre 0,5 e 2 e a referência fica fixa. Isso não é calibração radiométrica.

## 11. A costura busca um caminho de menor diferença (1.5 min)

Explicar mapa de erro, custo acumulado e retorno pelos predecessores. O caminho só passa em pixels da sobreposição. A nova vista fornece o lado direito. Seis costuras usaram programação dinâmica. O fallback por graph cut está documentado, mas não foi necessário nesta coleta.

## 12. Máscaras suaves combinam bandas em várias escalas (1.5 min)

Diferenciar pirâmide gaussiana de laplaciana. As máscaras têm pesos suaves em cada escala; as bandas são somadas e reconstruídas. Cinco níveis. A costura escolhe a fonte, enquanto o blending suaviza a transição. Mistura ainda pode aparecer perto das costuras.

## 13. O carro deixa de aparecer transparente (2 min)

Começar pelas duas fontes: uma tem carro e outra observa o fundo. Comparar média, feathering, costura e multibanda. A região foi anotada apenas para ilustração. Diferenças também podem vir de exposição e paralaxe. A métrica auxiliar mede mistura, não correção semântica.

## 14. O panorama preserva a varredura e expõe os limites (1.5 min)

Apontar no panorama exemplos de continuidade e defeitos locais. Não atribuir todos os problemas a movimento. O recorte elimina regiões sem fonte e remove parte do enquadramento alto. A comparação com Stitcher está no notebook e não isola um único parâmetro.

## 15. Cinco embaralhamentos mantiveram o resultado (1 min)

Descrever a geração de PNGs sem EXIF com os mesmos pixels de trabalho. A igualdade vale para esta coleta, não para cenas arbitrárias. Mostrar o comando de execução. Para uma nova coleta, priorizar luz diurna, pouca translação e maior sobreposição nas extremidades.

## 16. Métodos apoiados nos trabalhos originais (0.5 min)

Indicar os artigos centrais: Lowe para descritores, Brown e Lowe para reconhecimento e ajuste, Efros e Freeman para costura e Burt e Adelson para multibanda. Abrir para perguntas após fechar as limitações.
