"""Convertit les parquets en un dossier memmap : images décodées sur disque + métadonnées.

Lit un fichier parquet à la fois (~500 Mo), donc la mémoire reste faible même sur
Colab gratuit. Le résultat (~12,6 Go pour le jeu complet) se lit ensuite à la demande.

    python -m scripts.build_memmap --raw /content/raw --out /content/data/full
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import META_COLS  # noqa: E402

SIZE = 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="dossier des parquets, ou un fichier parquet")
    ap.add_argument("--out", default="data/full")
    args = ap.parse_args()

    raw = Path(args.raw)
    files = sorted(raw.glob("*.parquet")) if raw.is_dir() else [raw]
    n_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8, shape=(n_total, SIZE, SIZE, 3))
    metas, row, t0 = [], 0, time.time()
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=META_COLS + ["image"])
            pngs = t.column("image").combine_chunks().field("bytes").to_pylist()
            for k, b in enumerate(pngs):
                images[row + k] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            m = t.drop(["image"]).to_pandas()
            m["row"] = np.arange(row, row + len(m))
            metas.append(m)
            row += len(m)
        print(f"{f.name} : {row}/{n_total} patchs, {time.time() - t0:.0f} s", flush=True)
    images.flush()
    meta = pd.concat(metas, ignore_index=True)
    meta.to_parquet(out / "metadata.parquet", index=False)
    print(f"Terminé : {row} patchs -> {out} ({images.nbytes / 1e9:.1f} Go)")
    print(pd.crosstab(meta.center, meta.label, margins=True))


if __name__ == "__main__":
    main()
