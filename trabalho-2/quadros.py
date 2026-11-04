import json
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}
N_FRAMES = 100


def sharpness(frame):
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main(data, n=N_FRAMES):
    videos = sorted(p for p in (data / "video").glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS) if (data / "video").is_dir() else []
    out = data / "quadros"
    if not videos or out.is_dir():
        return
    out.mkdir()
    report = {}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        # One frame per window, the sharpest one: motion blur is the main failure of frames taken from a video.
        edges = np.linspace(0, total, n + 1).astype(int)
        best, k = [None] * n, 0
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            while k < n - 1 and i >= edges[k + 1]:
                k += 1
            score = sharpness(frame)
            if best[k] is None or score > best[k][0]:
                best[k] = (score, i, frame)
        cap.release()
        chosen = [b for b in best if b is not None]
        for score, i, frame in chosen:
            cv2.imwrite(str(out / f"{video.stem}_{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        report[video.name] = {"quadros_video": total, "fps": fps, "extraidos": len(chosen),
                              "nitidez_mediana": float(np.median([c[0] for c in chosen]))}
    (data / "quadros.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
