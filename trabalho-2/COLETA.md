# Coleta — T2, Variante B

A soma dos RAs do grupo termina em dígito ímpar → **Variante B** (objeto).

## Material

- Tabuleiro impresso em **100%** (nunca "ajustar à página") e colado em papelão ou foam board plano: `charuco_a4.pdf` (casas de 30 mm), `charuco_a3.pdf` (40 mm) ou `charuco_a2.pdf` (60 mm). Conferir a barra de 100 mm; medir com paquímetro 4 casas seguidas na horizontal e na vertical e anotar `quadrado_mm` = média / 4 (obrigatório em A3 e A2).
- Paquímetro; para o volume (extra): balança de cozinha, pote com água e linha fina.
- Fundo com textura em volta do tabuleiro (jornal, revista, tecido estampado).

## Objeto

- Pegada de até ~12 × 8 cm no A4, ~16 × 10 cm no A3 e ~24 × 16 cm no A2 (sobra ao menos uma fileira de casas visível em volta).
- Fosco, opaco, rígido, com textura e detalhe; nada de vidro, metal ou plástico brilhante.
- Resistente à água, se formos fazer o volume.
- Registrar com o professor antes da coleta.

## Câmera

- Fotos, não vídeo. Resolução padrão (12–16 MP), JPEG com EXIF; no iPhone, Ajustes → Câmera → Formatos → "Mais Compatível" (HEIC não é lido pelo pipeline).
- Sempre o mesmo aparelho e sempre na horizontal (paisagem), sem girar o celular entre fotos.
- Sem zoom durante toda a coleta; foco manual (ou AF travado); exposição manual fixa: ISO baixo, f/8 se der, obturador ≥ 1/125 s; balanço de branco fixo; sem HDR, modo retrato ou filtros. No celular: modo Pro, sempre a lente 1×.
- Luz difusa (dia nublado, janela sem sol direto, luz de teto), sem flash e sem sombras duras.
- Objeto e tabuleiro não se movem; quem anda é a câmera (sem mesa giratória).

## Trajetória (~110 fotos)

| pasta  | elevação        | fotos                    |
|--------|-----------------|--------------------------|
| `h1/`  | ~15° (baixa)    | 36, uma a cada ~10°      |
| `h2/`  | ~40° (média)    | 36                       |
| `h3/`  | ~65° (alta)     | 36                       |
| `topo/`| ~90°            | 4–6                      |
| `calib/` (opcional) | tabuleiro sozinho, inclinações e posições variadas, cantos da imagem inclusive | 15–20 |

- Distância constante em cada anel; objeto ocupando 50–70% do quadro e parte do tabuleiro sempre visível.
- Fotos em ordem ao redor do objeto, sem pular. A separação por altura permite a ablação (1, 2 ou 3 alturas; 36/18/12/9 fotos por anel) sem nova coleta.

## Medidas (`medidas.json`)

- Comprimento e largura (lados do menor retângulo que contém o objeto visto de cima) e altura acima da mesa: 3 leituras de cada.
- Volume (extra): pote com água na balança, tarar, mergulhar o objeto suspenso pela linha sem tocar fundo nem paredes; gramas = cm³. 3 leituras.
- Fotografar as medições para o relatório.

```json
{
  "dispositivo": "Olympus OM-D E-M5 Mark II, 14 mm",
  "quadrado_mm": 30.0,
  "comprimento_mm": [0, 0, 0],
  "largura_mm": [0, 0, 0],
  "altura_mm": [0, 0, 0],
  "volume_cm3": [0, 0, 0],
  "notas": "luz, horário, desafios"
}
```

## Entrega dos arquivos

`dados/<objeto>/{h1,h2,h3,topo,calib}/` + `medidas.json`, com os arquivos originais (cabo, Drive ou pendrive). Não mandar por WhatsApp: recomprime e apaga o EXIF.
