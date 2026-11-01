"""Lecture à distance des lames entières CAMELYON17 (BigTIFF pyramidaux, tuiles JPEG 512x512).

Les lames (~2 Go chacune) sont en accès libre sur AWS (s3://camelyon-dataset/CAMELYON17/images/).
Plutôt que de télécharger une lame entière, on lit l'en-tête TIFF puis seulement les tuiles JPEG
du niveau 2 qui recouvrent du tissu (quelques % du fichier), par requêtes HTTP « Range » parallèles.

Niveau 2 = sous-échantillonnage x4 (~0,97 µm/pixel) : même échelle que les patchs WILDS
(PATCH_LEVEL = 2 dans leur code de préparation). Une tuile de 96 px couvre donc ~93 µm.
"""
from __future__ import annotations

import re
import time

import fsspec
import numpy as np
import tifffile
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects

BASE_URL = "https://camelyon-dataset.s3.amazonaws.com/CAMELYON17"
PATCH_LEVEL, MASK_LEVEL, TILE = 2, 6, 96


class RemoteSlide:
    def __init__(self, name: str, base_url: str = BASE_URL):
        self.url = f"{base_url}/images/{name}"
        self.fs = fsspec.filesystem("http")
        self._fh = self.fs.open(self.url, block_size=2**20)
        self.tif = tifffile.TiffFile(self._fh)
        self.levels = self.tif.series[0].levels
        p0 = self.levels[0].pages[0]
        if "XResolution" in p0.tags:
            num, den = p0.tags["XResolution"].value
            unit = p0.tags["ResolutionUnit"].value  # 3 = centimètre
            mpp0 = 1 / (num / den / (1e4 if unit == 3 else 2.54e4))
        else:  # lames Philips : résolution dans la description XML (DICOM_PIXEL_SPACING, en mm)
            desc = p0.tags["ImageDescription"].value
            mpp0 = float(re.search(r'DICOM_PIXEL_SPACING"[^>]*>"([\d.]+)"', desc).group(1)) * 1e3
        self.shape0 = self.levels[0].shape[:2]
        self.page = self.levels[PATCH_LEVEL].pages[0]
        self.shape = self.page.shape[:2]  # (H, W) au niveau 2
        self.mpp = mpp0 * self.shape0[1] / self.shape[1]  # µm par pixel au niveau 2

    def close(self):
        self.tif.close()
        self._fh.close()

    def thumbnail(self, level: int = MASK_LEVEL) -> np.ndarray:
        return self.levels[min(level, len(self.levels) - 1)].asarray()  # certaines lames n'ont que 7 niveaux

    def tissue_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Masque tissu au niveau 6 : Otsu sur la saturation, fond blanc et noir exclus."""
        low = self.thumbnail()
        sat = rgb2hsv(low)[..., 1]
        bright = low.mean(-1)
        valid = (bright > 20) & (bright < 235)  # zones vides (noires) et fond blanc
        t = threshold_otsu(sat[valid]) if valid.any() else 1.0
        m = (sat > max(t, 0.05)) & valid
        m = remove_small_objects(binary_closing(m, disk(2)), 64)
        return m, low

    def tissue_tiles(self, min_frac: float = 0.25) -> np.ndarray:
        """Coordonnées (ligne, colonne) des tuiles 96x96 du niveau 2 contenant assez de tissu."""
        m, _ = self.tissue_mask()
        f = 2 ** (MASK_LEVEL - PATCH_LEVEL)  # 16 : une tuile de 96 px = 6 px au niveau 6
        k = TILE // f
        nr, nc = min(m.shape[0] // k, self.shape[0] // TILE), min(m.shape[1] // k, self.shape[1] // TILE)
        frac = m[:nr * k, :nc * k].reshape(nr, k, nc, k).mean((1, 3))
        return np.argwhere(frac >= min_frac)

    def read_tiles(self, grid: np.ndarray) -> np.ndarray:
        """Lit les tuiles 96x96 (uint8 NHWC) aux positions `grid` (ligne, colonne en unités de 96 px)."""
        p = self.page
        tw, tl = p.tilewidth, p.tilelength
        ntx = (self.shape[1] + tw - 1) // tw
        # tuiles JPEG nécessaires : celles qui recouvrent chaque tuile de 96 px
        need = set()
        for r, c in grid:
            y0, x0 = r * TILE, c * TILE
            for jy in range(y0 // tl, (y0 + TILE - 1) // tl + 1):
                for jx in range(x0 // tw, (x0 + TILE - 1) // tw + 1):
                    need.add((jy, jx))
        need = sorted(need)
        idx = [jy * ntx + jx for jy, jx in need]
        starts = [p.dataoffsets[i] for i in idx]
        ends = [p.dataoffsets[i] + p.databytecounts[i] for i in idx]
        # requêtes parallèles ; celles qui échouent (coupure réseau) sont refaites jusqu'à 5 fois
        blobs = self.fs.cat_ranges([self.url] * len(idx), starts, ends, batch_size=32, on_error="return")
        for attempt in range(5):
            bad = [k for k, b in enumerate(blobs) if not isinstance(b, bytes)]
            if not bad:
                break
            time.sleep(2 ** attempt)
            redo = self.fs.cat_ranges([self.url] * len(bad), [starts[k] for k in bad], [ends[k] for k in bad],
                                      batch_size=16, on_error="return")
            for k, b in zip(bad, redo):
                blobs[k] = b
        else:
            raise OSError(f"{self.url} : {len(bad)} tuiles JPEG illisibles après 5 essais")
        decode = p.decode
        cache = {}
        for key, i, data in zip(need, idx, blobs):
            seg, _, _ = decode(data, i, jpegtables=p.jpegtables)
            cache[key] = seg.reshape(tl, tw, -1)[..., :3]
        out = np.empty((len(grid), TILE, TILE, 3), np.uint8)
        for n, (r, c) in enumerate(grid):
            y0, x0 = r * TILE, c * TILE
            for jy in range(y0 // tl, (y0 + TILE - 1) // tl + 1):
                for jx in range(x0 // tw, (x0 + TILE - 1) // tw + 1):
                    ty, tx = jy * tl, jx * tw
                    ys, ye = max(y0, ty), min(y0 + TILE, ty + tl)
                    xs, xe = max(x0, tx), min(x0 + TILE, tx + tw)
                    out[n, ys - y0:ye - y0, xs - x0:xe - x0] = cache[(jy, jx)][ys - ty:ye - ty, xs - tx:xe - tx]
        return out
