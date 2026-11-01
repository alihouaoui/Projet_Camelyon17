"""Caractéristiques Phikon de toutes les tuiles de tissu des lames de la cohorte pN.

    python -m scripts.wsi_features --cohort configs/pn_cohort.csv --out /content/outputs/wsi

Pour chaque lame (ordre de configs/pn_cohort.csv) : masque tissu, lecture à distance des tuiles
96x96 du niveau 2, caractéristiques Phikon (CLS, 224 px). Une lame déjà traitée est sautée :
relancer après une coupure reprend à la lame suivante. La lame suivante est lue pendant que le GPU
travaille sur la lame courante.

Sortie par lame : <out>/<lame>.npz avec grid (ligne, colonne des tuiles), feats (float16),
mpp (µm/pixel au niveau 2), shape (taille du niveau 2), thumb (miniature niveau 7, pour les figures).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.foundation import Encoder  # noqa: E402
from src.wsi import RemoteSlide  # noqa: E402
from scripts.train_cnn import device  # noqa: E402


def load(name: str, max_tiles: int | None):
    """Lecture d'une lame (réseau + décodage JPEG), exécutée dans un fil séparé."""
    t0 = time.time()
    s = RemoteSlide(name)
    try:
        grid = s.tissue_tiles()
        if max_tiles:
            grid = grid[:max_tiles]
        tiles = np.concatenate([s.read_tiles(grid[i:i + 4096]) for i in range(0, len(grid), 4096)]) \
            if len(grid) else np.zeros((0, 96, 96, 3), np.uint8)
        thumb = s.thumbnail(7)
        return dict(grid=grid, tiles=tiles, mpp=s.mpp, shape=np.array(s.shape), thumb=thumb, t_read=time.time() - t0)
    finally:
        s.close()


def empty_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode(enc, tiles: np.ndarray, batch_size: int) -> np.ndarray:
    """Caractéristiques par lots ; si la mémoire GPU sature, on vide le cache et on divise le lot par 2."""
    feats, i = [], 0
    while i < len(tiles):
        try:
            feats.append(enc(torch.from_numpy(tiles[i:i + batch_size])).cpu().numpy().astype(np.float16))
            i += batch_size
        except RuntimeError as e:
            if "out of memory" not in str(e) or batch_size <= 8:
                raise
            empty_cache()
            batch_size //= 2
    return np.concatenate(feats) if feats else np.zeros((0, 768), np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="configs/pn_cohort.csv")
    ap.add_argument("--out", default="outputs/wsi")
    ap.add_argument("--model", default="phikon")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-slides", type=int, default=0)
    ap.add_argument("--max-tiles", type=int, default=0, help="test rapide : limite les tuiles par lame")
    ap.add_argument("--reverse", action="store_true", help="part de la fin de la cohorte (2e machine en parallèle)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slides = pd.read_csv(args.cohort).slide.tolist()
    todo = [s for s in slides if not (out / s.replace(".tif", ".npz")).exists()]
    n_done = len(slides) - len(todo)
    if args.reverse:
        todo = todo[::-1]
    if args.max_slides:
        todo = todo[:args.max_slides]
    dev = device()
    enc = Encoder(args.model, dev)
    print(f"{n_done} lames déjà faites, {len(todo)} à faire ({args.model} sur {dev})", flush=True)

    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(load, todo[0], args.max_tiles or None) if todo else None
        for k, name in enumerate(todo):
            try:
                d = nxt.result()
            except Exception as e:  # lame illisible (réseau) : on passe, elle sera reprise au prochain lancement
                print(f"Échec lame {name} (lecture impossible : {str(e)[:80]}), reprise au prochain lancement", flush=True)
                d = None
            nxt = pool.submit(load, todo[k + 1], args.max_tiles or None) if k + 1 < len(todo) else None
            if d is None:
                continue
            t0 = time.time()
            try:
                feats = encode(enc, d["tiles"], args.batch_size)
            except RuntimeError as e:  # mémoire GPU saturée malgré les réessais : lame reprise plus tard
                print(f"Échec lame {name} (GPU : {str(e)[:80]}), reprise au prochain lancement", flush=True)
                continue
            finally:
                del d["tiles"]
                empty_cache()
            path = out / name.replace(".tif", ".npz")
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, grid=d["grid"], feats=feats, mpp=d["mpp"], shape=d["shape"], thumb=d["thumb"])
            os.replace(tmp, path)
            print(f"Lame {n_done + k + 1}/{len(slides)} {name} : {len(feats)} tuiles, "
                  f"lecture {d['t_read']:.0f} s, GPU {time.time() - t0:.0f} s", flush=True)
    print(f"Terminé : {out}", flush=True)


if __name__ == "__main__":
    main()
