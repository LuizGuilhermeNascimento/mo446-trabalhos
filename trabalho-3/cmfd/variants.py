"""Variantes do detector usadas na ablação das ideias 1-3 (mesmas do teste sintético)."""
from dataclasses import replace

from .detector import Config

BASE = Config()

VARIANTS = {
    # ponto de partida "estilo Kaggle": só entre painéis, sem espelho, máscara retangular
    "kaggle_like": replace(BASE, panel_mode="inter", mirror=False, mask_mode="bbox"),
    "ideia_1": replace(BASE, panel_mode="all", mirror=False, mask_mode="bbox"),
    "ideias_1_2": replace(BASE, panel_mode="all", mirror=True, mask_mode="bbox"),
    "ideias_1_2_3hull": replace(BASE, panel_mode="all", mirror=True, mask_mode="hull"),
    "ideias_1_2_3holes": replace(BASE, panel_mode="all", mirror=True, mask_mode="dense", dense_fill="holes"),
    "ideias_1_2_3": replace(BASE, panel_mode="all", mirror=True, mask_mode="dense", dense_fill="hull"),
    # retirando um componente por vez da variante escolhida na validação (máscara densa com buracos fechados)
    "ideias_1_3": replace(BASE, panel_mode="all", mirror=False, mask_mode="dense", dense_fill="holes"),
    "ideias_2_3_inter": replace(BASE, panel_mode="inter", mirror=True, mask_mode="dense", dense_fill="holes"),
}
# Rumo a 0,8: mais pontos em imagens pequenas/lisas (ampliar a entrada + limiar de contraste mais baixo),
# como na literatura de CMFD para regiões pequenas e lisas. Base = ideias_1_3.
_B13 = VARIANTS["ideias_1_3"]
VARIANTS.update({
    "up1024": replace(_B13, min_side_upscale=1024),
    "up1024_ct002": replace(_B13, min_side_upscale=1024, contrast_threshold=0.002),
    "up1536_ct002_kp16k": replace(_B13, min_side_upscale=1536, contrast_threshold=0.002, max_keypoints=16000),
    "up2048_ct001_kp32k": replace(_B13, min_side_upscale=2048, contrast_threshold=0.001, max_keypoints=32000),
})
# + candidatos de um ViT pré-treinado e congelado (DINOv2-small), sem nenhum treino
VARIANTS["up1536_deep"] = replace(VARIANTS["up1536_ct002_kp16k"], deep=True)
# DINOv3 (pesos com licença da Meta via Hugging Face; patch 16 + 4 tokens de registro)
VARIANTS["up1536_deep3"] = replace(VARIANTS["up1536_deep"], deep_model="facebook/dinov3-vits16-pretrain-lvd1689m")

# campo denso de correspondências (Cozzolino et al. 2015) em cascata sobre a melhor base por SIFT
VARIANTS["up2048_field"] = replace(VARIANTS["up2048_ct001_kp32k"], field=True)
VARIANTS["up1536_field"] = replace(VARIANTS["up1536_ct002_kp16k"], field=True)

# sementes verificadas pelo ruído (pares isolados do SIFT)
VARIANTS["up1536_vseed"] = replace(VARIANTS["up1536_ct002_kp16k"], vseed=True)

# melhor base (SIFT ampliado até 2048) + sementes verificadas pelo ruído; máscara refeita depois com remask.py
VARIANTS["up2048_vseed"] = replace(VARIANTS["up2048_ct001_kp32k"], vseed=True)

# escalas extras para o detector multiescala (candidatos de várias escalas juntados por merge_caches.py)
VARIANTS["s1024"] = replace(VARIANTS["up1024_ct002"])
VARIANTS["s512"] = replace(VARIANTS["ideias_1_3"])

# piso do RANSAC 5 -> 3 e até 8 movimentos: só ACRESCENTA candidatos fracos (os de >= 5 inliers não mudam);
# quem decide se valem é o filtro de ruído + a seleção no treino
for _n, _b in [("f3_2048", "up2048_ct001_kp32k"), ("f3_1536", "up1536_ct002_kp16k"), ("f3_1024", "s1024"), ("f3_512", "s512")]:
    VARIANTS[_n] = replace(VARIANTS[_b], floor_inliers=3, max_models=8)

# uma passada só, no tamanho original da imagem (sem ampliar nem reduzir); demais ajustes do SIFT iguais aos da f3_2048
VARIANTS["nativo"] = replace(VARIANTS["f3_2048"], max_side=10**9, min_side_upscale=0)

# ----- pipeline alternativo: SEM SIFT, pares vindos só dos patches de um ViT congelado (mesmo RANSAC, máscara e
# filtros). DINOv3 tem patch de 16 px: 768 px no lado maior = 48 patches; 1024 px = 64 patches.
_DEEP = replace(VARIANTS["up1536_ct002_kp16k"], points=False, deep=True, deep_cascade=False)
_D3 = "facebook/dinov3-vit{}16-pretrain-lvd1689m"
VARIANTS.update({
    "d3s_768": replace(_DEEP, deep_model=_D3.format("s"), deep_target=768),
    "d3s_1024": replace(_DEEP, deep_model=_D3.format("s"), deep_target=1024),
    "d3b_768": replace(_DEEP, deep_model=_D3.format("b"), deep_target=768),
    "d3b_1024": replace(_DEEP, deep_model=_D3.format("b"), deep_target=1024),
})
