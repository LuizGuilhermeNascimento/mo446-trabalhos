from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SQUARES, SQUARE_MM, MARKER_MM = (9, 6), 30, 22
PAPERS = {"a4": (297, 210, 30), "a3": (420, 297, 40), "a2": (594, 420, 60)}
BOARD = cv2.aruco.CharucoBoard(SQUARES, SQUARE_MM, MARKER_MM, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
DETECTOR = cv2.aruco.CharucoDetector(BOARD)


def detect(gray):
    corners, ids, _, _ = DETECTOR.detectBoard(gray)
    if ids is None:
        return np.empty((0, 2), np.float32), np.empty(0, int)
    return corners.reshape(-1, 2), ids.ravel()


def page(paper="a4", px_per_mm=10):
    width_mm, height_mm, square = PAPERS[paper]
    sheet = np.full((height_mm * px_per_mm, width_mm * px_per_mm), 255, np.uint8)
    w, h = (n * square * px_per_mm for n in SQUARES)
    x0, y0 = (sheet.shape[1] - w) // 2, (sheet.shape[0] - h) // 2
    sheet[y0:y0 + h, x0:x0 + w] = BOARD.generateImage((w, h))
    y = y0 // 2
    cv2.line(sheet, (x0, y), (x0 + 100 * px_per_mm, y), 0, 3)
    for mm in range(0, 101, 10):
        cv2.line(sheet, (x0 + mm * px_per_mm, y - 15), (x0 + mm * px_per_mm, y + 15), 0, 3)
    marker = square * MARKER_MM / SQUARE_MM
    text = (f"ChArUco {SQUARES[0]}x{SQUARES[1]} DICT_5X5_100 {paper.upper()} | quadrado {square} mm | marcador {marker:g} mm"
            f" | imprimir em 100% | barra = 100 mm | anotar quadrado_mm medido")
    cv2.putText(sheet, text, (x0, y0 + h + (sheet.shape[0] - y0 - h) // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    return sheet


if __name__ == "__main__":
    for paper in PAPERS:
        out = Path(__file__).with_name(f"charuco_{paper}.pdf")
        Image.fromarray(page(paper)).save(out, resolution=254)
        print(out)
