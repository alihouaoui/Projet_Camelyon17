"""Construit un sous-échantillon représentatif sans télécharger les 10,7 Go.

Les parquets du miroir sont triés (split, puis classe, puis lame) : un seul fichier ne
contient qu'une partie des centres et des classes. On lit donc à distance quelques
groupes de lignes (100 patchs chacun) régulièrement espacés dans chacun des 21 fichiers.

    python -m scripts.fetch_sample --groups-per-file 20 --out data/processed/subset.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import META_COLS  # noqa: E402

REPO = "datasets/wltjr1007/Camelyon17-WILDS/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-per-file", type=int, default=20)
    ap.add_argument("--out", default="data/processed/subset.parquet")
    args = ap.parse_args()

    fs = HfFileSystem()
    files = sorted(fs.ls(REPO, detail=False))
    files = [f for f in files if f.endswith(".parquet")]
    tables = []
    for f in files:
        with fs.open(f, "rb", block_size=4 * 2**20) as fh:
            pf = pq.ParquetFile(fh)
            n = pf.metadata.num_row_groups
            groups = np.unique(np.linspace(0, n - 1, min(n, args.groups_per_file)).round().astype(int)).tolist()
            t = pf.read_row_groups(groups, columns=META_COLS + ["image"])
        tables.append(t)
        print(f"{Path(f).name}: {t.num_rows} patchs ({len(groups)}/{n} groupes)")
    t = pa.concat_tables(tables)
    img = t.column("image").combine_chunks().field("bytes")
    df = t.drop(["image"]).to_pandas()
    df["image"] = [{"bytes": b, "path": None} for b in img.to_pylist()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} patchs -> {args.out}")
    print(pd.crosstab(df.center, df.label, margins=True))


if __name__ == "__main__":
    main()
