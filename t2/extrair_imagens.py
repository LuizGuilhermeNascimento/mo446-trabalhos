"""Extrai quadros uniformemente espaçados de um arquivo de vídeo."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
VIDEO_PADRAO = ROOT / "images" / "pedra_cubica" / "P1011579.MOV"
DESTINO_PADRAO = ROOT / "images" / "pedra_cubica"


def extrair_imagens(
    video_path: str | Path,
    output_dir: str | Path = DESTINO_PADRAO,
    num_frames: int = 80,
) -> list[Path]:
    """Recebe o caminho do vídeo e salva quadros distribuídos ao longo dele.

    Os nomes incluem a posição e o tempo do quadro. Reexecutar com a mesma
    quantidade regenera os mesmos nomes de arquivo.
    """
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")
    if num_frames < 1:
        raise ValueError("num_frames precisa ser maior que zero")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if total_frames < 1:
            raise RuntimeError("O vídeo não informa nenhum quadro para extração.")

        frame_indices = np.linspace(
            0, total_frames - 1, num=min(num_frames, total_frames), dtype=int
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for sequence, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                continue

            timestamp = frame_index / fps if fps > 0 else 0.0
            image_path = output_dir / f"quadro_{sequence:04d}_de_{len(frame_indices):04d}_t{timestamp:08.2f}s.jpg"
            if not cv2.imwrite(
                str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Não foi possível salvar a imagem: {image_path}")
            saved_paths.append(image_path)

        if not saved_paths:
            raise RuntimeError("Nenhum quadro pôde ser lido do vídeo.")
        return saved_paths
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path, default=VIDEO_PADRAO,
        help=f"Arquivo de vídeo (padrão: {VIDEO_PADRAO.name})",
    )
    parser.add_argument("--frames", type=int, default=80, help="Número de quadros (padrão: 80)")
    parser.add_argument("--saida", type=Path, default=DESTINO_PADRAO)
    args = parser.parse_args()

    paths = extrair_imagens(args.video, args.saida, args.frames)
    print(f"Vídeo: {args.video.resolve()}")
    print(f"Quadros salvos: {len(paths)} em {args.saida.resolve()}")
    if len(paths) < 40:
        print("Aviso: foram extraídos menos de 40 quadros; confira a duração do vídeo.")


if __name__ == "__main__":
    main()
