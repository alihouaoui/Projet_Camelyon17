"""Augmentations. Les patchs histologiques n'ont pas d'orientation : rotations et
symétries sont donc sans risque. L'augmentation HED perturbe la coloration H&E pour
simuler la variabilité entre hôpitaux (Tellez et al., 2019)."""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import hed2rgb, rgb2hed
import torch
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HEDJitter:
    """Multiplie et décale chaque canal H, E, D d'un facteur aléatoire.

    Utilise l'aléa global de numpy, réinitialisé par PyTorch dans chaque worker.
    """

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        hed = rgb2hed(np.asarray(img, dtype=np.float32) / 255.0)
        alpha = np.random.uniform(1 - self.sigma, 1 + self.sigma, size=3)
        beta = np.random.uniform(-self.sigma, self.sigma, size=3)
        rgb = np.clip(hed2rgb(hed * alpha + beta), 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8))


class RandomRot90:
    def __call__(self, img: Image.Image) -> Image.Image:
        k = np.random.randint(4)
        return img.rotate(90 * k) if k else img


def train_transform(hed: bool = False, size: int = 96):
    ops = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), RandomRot90()]
    if hed:
        ops.append(HEDJitter())
    if size != 96:
        ops.append(T.Resize(size))
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def eval_transform(size: int = 96):
    ops = [T.Resize(size)] if size != 96 else []
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def hed_jitter_batch(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """HEDJitter par lot sur le GPU (x : float NCHW dans [0, 1]).

    Même calcul que skimage rgb2hed / hed2rgb (déconvolution de Ruifrok), avec un alpha
    et un beta tirés par image et par canal, comme HEDJitter.
    """
    from skimage.color.colorconv import hed_from_rgb, rgb_from_hed
    n = x.shape[0]
    to_hed = torch.tensor(hed_from_rgb, dtype=x.dtype, device=x.device)
    to_rgb = torch.tensor(rgb_from_hed, dtype=x.dtype, device=x.device)
    log_adjust = float(np.log(1e-6))
    rgb = x.permute(0, 2, 3, 1).clamp_min(1e-6)                      # NHWC
    hed = ((rgb.log() / log_adjust) @ to_hed).clamp_min(0)
    alpha = torch.empty(n, 1, 1, 3, device=x.device, dtype=x.dtype).uniform_(1 - sigma, 1 + sigma)
    beta = torch.empty(n, 1, 1, 3, device=x.device, dtype=x.dtype).uniform_(-sigma, sigma)
    hed = hed * alpha + beta
    rgb = torch.exp(-(hed * -log_adjust) @ to_rgb).clamp(0, 1)
    return rgb.permute(0, 3, 1, 2).contiguous()


def gpu_prepare(x: torch.Tensor, train: bool, hed: bool = False) -> torch.Tensor:
    """Préparation par lot sur le GPU : uint8 NHWC -> float NCHW normalisé.

    En entraînement, chaque patch reçoit une symétrie et une rotation de 90° aléatoires
    (mêmes augmentations que train_transform sans HED). Sur Colab gratuit (2 cœurs CPU),
    faire ces opérations image par image avec PIL limitait la vitesse d'entraînement.
    """
    x = x.permute(0, 3, 1, 2).float().div_(255)
    if train:
        n = x.shape[0]
        flip_h = torch.rand(n, device=x.device) < 0.5
        x = torch.where(flip_h[:, None, None, None], x.flip(3), x)
        flip_v = torch.rand(n, device=x.device) < 0.5
        x = torch.where(flip_v[:, None, None, None], x.flip(2), x)
        k = torch.randint(0, 4, (n,), device=x.device)
        for r in (1, 2, 3):
            m = k == r
            if m.any():
                x[m] = torch.rot90(x[m], r, (2, 3))
        if hed:
            x = hed_jitter_batch(x)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=x.device)[None, :, None, None]
    return (x - mean) / std
