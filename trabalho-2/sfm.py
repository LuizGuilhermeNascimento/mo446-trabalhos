import os
import sys
from pathlib import Path

import pycolmap

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_names(data):
    return sorted(str(p.relative_to(data)) for p in data.rglob("*")
                  if p.suffix.lower() in EXTENSIONS and p.relative_to(data).parts[0] != "calib")


def main(data, out):
    out.mkdir(parents=True, exist_ok=True)
    database = out / "database.db"
    database.unlink(missing_ok=True)
    reader = pycolmap.ImageReaderOptions(camera_model="RADIAL")
    # CPU SIFT needs up to ~2 GB per thread on 9-12 MP photos; all 12 cores ran a 16 GB laptop out of memory.
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    threads = max(1, min(os.cpu_count(), int(ram_gb // 2)))
    extraction = pycolmap.FeatureExtractionOptions(max_image_size=3200, num_threads=threads)
    pycolmap.extract_features(database, data, image_names(data), pycolmap.CameraMode.SINGLE, reader, extraction)
    pycolmap.match_exhaustive(database)
    models = pycolmap.incremental_mapping(database, data, out / "modelos")
    best = max(models.values(), key=lambda r: r.num_reg_images())
    (out / "esparso").mkdir(exist_ok=True)
    best.write(out / "esparso")
    print(best.summary())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
