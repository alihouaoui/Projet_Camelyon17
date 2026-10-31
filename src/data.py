"""Chargement des patchs Camelyon17-WILDS, découpage sans fuite et Dataset PyTorch.

Source : miroir Hugging Face `wltjr1007/Camelyon17-WILDS` (fichiers parquet), qui
reprend les 455 954 patchs 96x96 de WILDS avec leurs métadonnées.
Centres : 0, 3, 4 = entraînement ; 1 = validation hors domaine ; 2 = test hors domaine.

Deux formats de stockage :
- parquet (octets PNG en mémoire) : pratique pour l'échantillon local ;
- dossier « memmap » (`images.npy` + `metadata.parquet`, voir scripts/build_memmap.py) :
  images décodées sur disque, lues à la demande. Indispensable pour le jeu complet
  sur Colab gratuit (12,7 Go de RAM pour ~12,6 Go d'images décodées).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

META_COLS = ["image_id", "patient", "node", "slide", "center", "x_coord", "y_coord", "label"]
TRAIN_CENTERS = (0, 3, 4)
VAL_OOD_CENTER = 1
TEST_OOD_CENTER = 2


def load_parquet(files: list[str | Path], with_images: bool = True) -> pd.DataFrame:
    """Lit une liste de fichiers parquet. Les images restent en octets PNG (colonne `img`)."""
    frames = []
    for f in files:
        cols = META_COLS + (["image"] if with_images else [])
        table = pq.read_table(f, columns=cols)
        df = table.drop(["image"]).to_pandas() if with_images else table.to_pandas()
        if with_images:
            df["img"] = table.column("image").combine_chunks().field("bytes").to_pylist()
        df["source_file"] = Path(f).name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["patient_id"] = df["patient"].astype(int)  # identifiant patiente unique dans Camelyon17
    df["slide_id"] = df["patient"].astype(str) + "_" + df["node"].astype(str)
    return df


def load_data(spec: str | Path) -> tuple[pd.DataFrame, Path | None]:
    """Charge un dossier memmap ou un motif de fichiers parquet.

    Renvoie (métadonnées, chemin de images.npy). Pour le format parquet, le chemin vaut
    None et les octets PNG sont dans la colonne `img`.
    """
    spec = Path(spec)
    if spec.is_dir() and (spec / "images.npy").exists():
        df = pd.read_parquet(spec / "metadata.parquet")
        df["patient_id"] = df["patient"].astype(int)
        df["slide_id"] = df["patient"].astype(str) + "_" + df["node"].astype(str)
        return df, spec / "images.npy"
    files = sorted(Path().glob(str(spec))) if not spec.is_absolute() else sorted(spec.parent.glob(spec.name))
    if not files:
        raise FileNotFoundError(f"Aucune donnée trouvée pour {spec}")
    return load_parquet(files), None


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Écrit au même format que les parquets sources (colonne struct `image`)."""
    out = df[META_COLS].copy()
    out["image"] = [{"bytes": b, "path": None} for b in df["img"]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def assign_splits(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Découpage par patiente (jamais par patch) pour éviter la fuite de données.

    - Centres d'entraînement : une fraction des patientes de chaque centre part en `val_id`.
    - Centre 1 -> `val_ood`, centre 2 -> `test_ood` (aucune patiente partagée par construction).

    Note : le split `id_val` officiel de WILDS tire des patchs des mêmes lames que
    l'entraînement ; on ne l'utilise pas pour cette raison.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "train"
    for c in TRAIN_CENTERS:
        patients = np.sort(df.loc[df.center == c, "patient_id"].unique())
        if len(patients) < 2:
            continue
        n_val = max(1, int(round(val_frac * len(patients))))
        val_patients = rng.choice(patients, size=n_val, replace=False)
        df.loc[(df.center == c) & df.patient_id.isin(val_patients), "split"] = "val_id"
    df.loc[df.center == VAL_OOD_CENTER, "split"] = "val_ood"
    df.loc[df.center == TEST_OOD_CENTER, "split"] = "test_ood"
    check_no_leakage(df)
    return df


def subsample(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Sous-échantillonne en gardant la proportion de chaque (lame, classe)."""
    if n is None or len(df) <= n:
        return df
    return df.groupby(["slide_id", "label"], group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(n * len(g) / len(df)))), random_state=seed))


def check_no_leakage(df: pd.DataFrame) -> None:
    """Vérifie qu'aucune patiente n'apparaît dans deux splits."""
    per_patient = df.groupby("patient_id")["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):
        raise ValueError(f"Fuite : patientes présentes dans plusieurs splits : {list(leaked.index)}")


def decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


class ImageSource:
    """Accès uniforme aux images, en octets PNG (colonne `img`) ou dans images.npy.

    Le memmap est ouvert à la première lecture, donc séparément dans chaque worker
    du DataLoader : rien n'est copié en mémoire au moment du fork ou du pickle.
    """

    def __init__(self, df: pd.DataFrame, images_path: Path | None = None):
        self.images_path = images_path
        self.items = df["row"].to_numpy() if images_path is not None else df["img"].tolist()
        self._mm = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i) -> np.ndarray:
        if self.images_path is None:
            return decode(self.items[i])
        if self._mm is None:
            self._mm = np.load(self.images_path, mmap_mode="r")
        return np.asarray(self._mm[self.items[i]])

    def __getstate__(self):
        return {**self.__dict__, "_mm": None}

    def slice(self, start: int, stop: int) -> "ImageSource":
        """Sous-ensemble léger (pour envoyer un paquet d'images à un processus)."""
        sub = object.__new__(ImageSource)
        sub.images_path, sub.items, sub._mm = self.images_path, self.items[start:stop], None
        return sub


class PatchDataset:
    """Dataset PyTorch minimal : lit l'image (PNG ou memmap) et applique `transform` (None = tableau brut)."""

    def __init__(self, df: pd.DataFrame, transform=None, images_path: Path | None = None):
        self.src = ImageSource(df, images_path)
        self.labels = df["label"].to_numpy().astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        if self.transform is None:  # tableau uint8 HWC brut, préparé ensuite par lot sur le GPU
            return np.ascontiguousarray(self.src[i]), self.labels[i]
        return self.transform(Image.fromarray(self.src[i])), self.labels[i]
