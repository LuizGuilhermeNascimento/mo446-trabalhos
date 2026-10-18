"""Executa a preparação das imagens da Variante B (etapa 1 do T2).

Este é o ponto de entrada para a etapa de dados. A reconstrução SfM/MVS ainda
deve ser adicionada quando os respectivos passos do pipeline forem definidos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extrair_imagens import DESTINO_PADRAO, VIDEO_PADRAO, extrair_imagens


def listar_imagens(pasta: Path) -> list[Path]:
    extensoes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(
        arquivo for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO_PADRAO)
    parser.add_argument("--imagens", type=Path, default=DESTINO_PADRAO)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument(
        "--usar-imagens-existentes",
        action="store_true",
        help="Não extrai o vídeo; usa imagens que já estão em --imagens.",
    )
    args = parser.parse_args()

    pasta_imagens = args.imagens.expanduser().resolve()
    if args.usar_imagens_existentes:
        imagens = listar_imagens(pasta_imagens) if pasta_imagens.is_dir() else []
        if not imagens:
            parser.error(f"Nenhuma imagem encontrada em {pasta_imagens}")
        video = None
    else:
        video = args.video.expanduser().resolve()
        imagens = extrair_imagens(video, pasta_imagens, args.frames)

    resumo = {
        "variante": "B",
        "etapa_concluida": "coleta_e_preparacao_das_imagens",
        "video": str(video) if video else None,
        "pasta_imagens": str(pasta_imagens),
        "quantidade_imagens": len(imagens),
        "sfm_mvs": "ainda nao implementado",
    }
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
