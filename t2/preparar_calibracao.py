"""Prepara quadros e calibração do grid antes da reconstrução principal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibracao_grade import calibrar_camera_papel_quadriculado
from extrair_imagens import extrair_imagens


RAIZ = Path(__file__).resolve().parent
VIDEO_PADRAO = RAIZ / "images" / "pedra_cubica" / "P1011579.MOV"
PASTA_QUADROS_PADRAO = RAIZ / "images" / "pedra_cubica"


def preparar(
    *,
    video: Path,
    pasta_quadros: Path,
    quantidade_quadros: int,
    identificador: str,
    objeto: str,
    indices_calibracao: tuple[int, ...] | None,
    refazer_quadros: bool = False,
    refazer_calibracao: bool = False,
) -> Path:
    """Extrai/reutiliza quadros e grava a calibração para a pipeline principal."""
    pasta_quadros.mkdir(parents=True, exist_ok=True)
    quadros = sorted(pasta_quadros.glob(f"quadro_*_de_{quantidade_quadros:04d}_*.jpg"))
    if quantidade_quadros == 80 and len(quadros) != quantidade_quadros:
        quadros = sorted(pasta_quadros.glob("frame_*.jpg"))
    if refazer_quadros or len(quadros) != quantidade_quadros:
        quadros = extrair_imagens(video, pasta_quadros, quantidade_quadros)
    if len(quadros) < 5:
        raise RuntimeError("A calibração requer pelo menos cinco quadros.")
    if indices_calibracao is None:
        indices_calibracao = tuple(range(max(0, len(quadros) - 7), len(quadros)))
    if any(indice < 0 or indice >= len(quadros) for indice in indices_calibracao):
        raise ValueError("Os índices de calibração ultrapassam a quantidade de quadros disponível.")

    caminhos_calibracao = [quadros[indice] for indice in indices_calibracao]
    tamanho_grade = (5, 5)
    tamanho_quadrado_m = 0.005
    marca_quadros = f"{len(quadros)}quadros"
    pasta_saida = RAIZ / "outputs" / objeto / f"{identificador}_{marca_quadros}_calibracao"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_cache = pasta_saida / f"calibracao_camera_{marca_quadros}.npz"
    caminho_metadados = pasta_saida / f"calibracao_camera_{marca_quadros}.json"
    configuracao = {
        "quadros": [caminho.name for caminho in caminhos_calibracao],
        "grid_size": list(tamanho_grade),
        "square_size_m": tamanho_quadrado_m,
    }

    resultado = None
    if caminho_cache.is_file() and caminho_metadados.is_file() and not refazer_calibracao:
        metadados_antigos = json.loads(caminho_metadados.read_text(encoding="utf-8"))
        if metadados_antigos.get("settings") == configuracao:
            with np.load(caminho_cache) as dados:
                resultado = {
                    "camera_matrix": dados["camera_matrix"],
                    "dist_coeffs": dados["dist_coeffs"],
                    "rms_px": float(dados["rms_px"]),
                }
            print(f"Calibração existente reutilizada: {caminho_cache}")

    if resultado is None:
        print(f"Quadros disponíveis: {len(quadros)}. A seleção abrirá uma janela para cada vista.")
        for caminho in caminhos_calibracao:
            print(f"Vista de calibração: {caminho.name}")
        resultado = calibrar_camera_papel_quadriculado(
            caminhos_calibracao,
            grid_size=tamanho_grade,
            square_size_m=tamanho_quadrado_m,
        )
        np.savez_compressed(
            caminho_cache,
            camera_matrix=resultado["camera_matrix"],
            dist_coeffs=resultado["dist_coeffs"],
            rms_px=resultado["rms_px"],
            per_view_rms_px=resultado["per_view_rms_px"],
            image_size=resultado["image_size"],
            image_points=np.stack(resultado["image_points"]),
        )
        caminho_metadados.write_text(
            json.dumps(
                {
                    "settings": configuracao,
                    "rms_px": resultado["rms_px"],
                    "per_view_rms_px": resultado["per_view_rms_px"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        for caminho, sobreposicao in zip(caminhos_calibracao, resultado["overlays"]):
            cv2.imwrite(str(pasta_saida / f"{caminho.stem}_sobreposicao_grade.jpg"), sobreposicao)

    print(f"Calibração pronta: {caminho_cache}")
    print(f"Erro RMS: {resultado['rms_px']:.3f} px")
    return caminho_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO_PADRAO)
    parser.add_argument("--pasta-quadros", type=Path, default=PASTA_QUADROS_PADRAO)
    parser.add_argument("--quadros", type=int, default=80)
    parser.add_argument("--captura", default="pedra_cubica_base")
    parser.add_argument("--objeto", default="pedra_cubica")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Índices de calibração; por padrão, usa os sete quadros finais.")
    parser.add_argument("--refazer-quadros", action="store_true")
    parser.add_argument("--refazer-calibracao", action="store_true")
    args = parser.parse_args()
    preparar(
        video=args.video.resolve(),
        pasta_quadros=args.pasta_quadros.resolve(),
        quantidade_quadros=args.quadros,
        identificador=args.captura,
        objeto=args.objeto,
        indices_calibracao=tuple(args.indices) if args.indices is not None else None,
        refazer_quadros=args.refazer_quadros,
        refazer_calibracao=args.refazer_calibracao,
    )


if __name__ == "__main__":
    main()
