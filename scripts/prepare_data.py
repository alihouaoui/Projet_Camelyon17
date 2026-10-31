"""Construit un sous-échantillon local (images incluses) à partir des parquets bruts.

Exemple :
    python -m scripts.prepare_data --raw data/raw --out data/processed/subset.parquet --per-slide 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_parquet, save_parquet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/subset.parquet")
    ap.add_argument("--per-slide", type=int, default=400, help="patchs max par (lame, classe)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.raw).glob("*.parquet"))
    df = load_parquet(files)
    print(f"{len(df)} patchs lus depuis {len(files)} fichiers")
    sub = (df.groupby(["slide_id", "label"], group_keys=False)
             .apply(lambda g: g.sample(min(len(g), args.per_slide), random_state=args.seed)))
    save_parquet(sub, args.out)
    print(f"{len(sub)} patchs -> {args.out}")
    print(sub.groupby(["center", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
