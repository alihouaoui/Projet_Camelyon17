"""Extrait les caractéristiques d'un modèle de fondation figé, par morceaux reprenables.

Exemples :
    # test local sur le sous-échantillon
    python -m scripts.extract_features --model phikon --data data/subset_mm --max-rows 2000
    # Colab, jeu complet
    python -m scripts.extract_features --model phikon_v2 --data /content/data/full --out /content/outputs/features

Sorties dans <out>/<model>/ : index.parquet (métadonnées des patchs retenus, dans l'ordre),
feats_0000.npy, feats_0001.npy... (float16). Un morceau déjà écrit est sauté : relancer après
une coupure reprend là où on s'était arrêté. Le découpage train/val/test n'est PAS figé ici :
il est refait au moment de la sonde (scripts/linear_probe.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ImageSource, load_data, subsample  # noqa: E402
from src.foundation import Encoder  # noqa: E402
from scripts.train_cnn import device  # noqa: E402

INDEX_COLS = ["image_id", "center", "patient", "node", "slide_id", "x_coord", "y_coord", "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/subset_mm")
    ap.add_argument("--out", default="outputs/features")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = tous les patchs ; sinon tirage stratifié (lame, classe)")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    df, images_path = load_data(args.data)
    df = subsample(df, args.max_rows or None, args.seed)
    df = df.sort_values("row" if "row" in df else "image_id").reset_index(drop=True)  # lecture séquentielle du memmap
    index = df[INDEX_COLS + (["row"] if "row" in df else [])]
    if (out / "index.parquet").exists():
        prev = pd.read_parquet(out / "index.parquet")
        if not prev.image_id.equals(index.image_id):
            raise ValueError(f"{out} contient une extraction faite sur d'autres patchs ; changer --out")
    else:
        index.to_parquet(out / "index.parquet", index=False)

    dev = device()
    enc = Encoder(args.model, dev)
    src = ImageSource(df, images_path)
    n_chunks = (len(df) + args.chunk - 1) // args.chunk
    print(f"{args.model} sur {dev} : {len(df)} patchs, {n_chunks} morceaux", flush=True)
    for k in range(n_chunks):
        path = out / f"feats_{k:04d}.npy"
        if path.exists():
            continue
        t0, feats = time.time(), []
        lo, hi = k * args.chunk, min((k + 1) * args.chunk, len(df))
        for b in range(lo, hi, args.batch_size):
            part = src.slice(b, min(b + args.batch_size, hi))
            x = torch.from_numpy(np.stack([part[i] for i in range(len(part))]))
            if dev.type == "cuda":
                x = x.pin_memory()
            feats.append(enc(x).cpu().numpy().astype(np.float16))
        feats = np.concatenate(feats)
        if not np.isfinite(feats).all():
            raise FloatingPointError(f"valeurs non finies dans le morceau {k} (fp16)")
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, feats)
        os.replace(tmp, path)  # écriture atomique : un morceau présent est toujours complet
        dt = time.time() - t0
        print(f"Morceau {k + 1}/{n_chunks} : {hi - lo} patchs en {dt:.0f} s ({(hi - lo) / dt:.0f} img/s), "
              f"dim {feats.shape[1]}", flush=True)
    print(f"Terminé : {out}", flush=True)


if __name__ == "__main__":
    main()
