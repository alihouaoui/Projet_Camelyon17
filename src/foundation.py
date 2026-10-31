"""Modèles de fondation en pathologie, utilisés figés comme extracteurs de caractéristiques.

Tous sont téléchargeables sans jeton Hugging Face. Aucun n'a CAMELYON dans ses données
de pré-entraînement déclarées (voir les model cards) :
- phikon      : ViT-B/16, iBOT, 40 M tuiles TCGA (Owkin) ;
- phikon_v2   : ViT-L/16, DINOv2, 456 M tuiles TCGA + CPTAC + GTEx (Owkin) ;
- midnight    : ViT-g/14, DINOv2 modifié, 12 k lames TCGA (kaiko.ai, Midnight-12k) ;
- dinov2_base : ViT-B/14 DINOv2 générique (images naturelles), contrôle sans pathologie.

Les patchs WILDS (96x96, ~1 µm/pixel) sont agrandis à 224x224 sur le GPU, taille de
pré-entraînement des modèles : même champ de vue, mais moins de détail réel qu'une tuile à 20x.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# pool : "cls" (token CLS) ou "cls_mean" (CLS concaténé à la moyenne des tokens, conseillé par kaiko)
MODELS = {
    "phikon": dict(hf="owkin/phikon", norm=IMAGENET, pool="cls"),
    "phikon_v2": dict(hf="owkin/phikon-v2", norm=IMAGENET, pool="cls"),
    "midnight": dict(hf="kaiko-ai/midnight", norm=HALF, pool="cls_mean"),
    "dinov2_base": dict(hf="facebook/dinov2-base", norm=IMAGENET, pool="cls"),
}


class Encoder:
    """Charge un modèle et transforme un lot uint8 NHWC (96x96) en caractéristiques float32."""

    def __init__(self, name: str, dev: torch.device, size: int = 224):
        from transformers import AutoModel

        spec = MODELS[name]
        self.dev, self.size, self.pool = dev, size, spec["pool"]
        # poids directement en fp16 sur GPU (Midnight : 2,2 Go au lieu de 4,4 Go)
        self.dtype = torch.float16 if dev.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(spec["hf"], dtype=self.dtype, low_cpu_mem_usage=True,
                                               attn_implementation="sdpa").to(dev).eval()
        self.mean = torch.tensor(spec["norm"][0], device=dev)[None, :, None, None]
        self.std = torch.tensor(spec["norm"][1], device=dev)[None, :, None, None]

    @torch.inference_mode()
    def __call__(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        mode = "bilinear" if self.dev.type == "mps" else "bicubic"  # bicubique absent sur MPS (Mac)
        x = F.interpolate(x, size=(self.size, self.size), mode=mode, align_corners=False).clamp_(0, 1)
        x = ((x - self.mean) / self.std).to(self.dtype)
        h = self.model(pixel_values=x).last_hidden_state
        f = h[:, 0] if self.pool == "cls" else torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
        return f.float()
