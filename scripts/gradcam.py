"""Grad-CAM du ResNet18 sur l'hôpital de test (C2) : où regarde le modèle ?

    python -m scripts.gradcam [--weights outputs_colab/outputs/resnet18_full/best.pt] [--tag ref]

Patchs C2 du sous-échantillon local (data/subset_mm). On montre les tumeurs détectées les plus sûres,
les tumeurs ratées les plus franches (au seuil appris sur val_id) et les fausses alertes.
Grad-CAM (Selvaraju et al., 2017) sur layer3 (carte 6×6 pour un patch 96×96, plus fine que layer4 en 3×3).
Lecture prudente : Grad-CAM montre une corrélation avec la décision, pas une explication causale.
Sortie : reports/figures/gradcam_c2_<tag>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.transforms import gpu_prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs_colab/outputs/resnet18_full/best.pt")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    metrics = Path(a.metrics or Path(a.weights).parent / "metrics.csv")
    m = pd.read_csv(metrics)
    thr = float(m[m.rule.str.startswith("sens>=")].threshold.iloc[0])

    df, img_path = load_data("data/subset_mm")
    imgs = np.load(img_path, mmap_mode="r")
    c2 = df[df.center == 2].sample(min(a.n, int((df.center == 2).sum())), random_state=0)
    model = build_model("resnet18", pretrained=False)
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()

    rows = c2.index.to_numpy()
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            x = gpu_prepare(torch.from_numpy(np.ascontiguousarray(imgs[rows[i:i + 256]])), train=False)
            probs.append(F.softmax(model(x), 1)[:, 1].numpy())
    c2 = c2.assign(prob=np.concatenate(probs))
    tp = c2[(c2.label == 1) & (c2.prob >= thr)].nlargest(4, "prob")
    fn = c2[(c2.label == 1) & (c2.prob < thr)].nsmallest(4, "prob")
    fp = c2[(c2.label == 0) & (c2.prob >= thr)].nlargest(4, "prob")
    print(f"seuil {thr:.3f} | C2 échantillon : {len(c2)} patchs, sensibilité {(c2[c2.label == 1].prob >= thr).mean():.2f}")

    feats, grads = {}, {}
    layer = model.layer3
    layer.register_forward_hook(lambda m_, i, o: feats.__setitem__("a", o))
    layer.register_full_backward_hook(lambda m_, gi, go: grads.__setitem__("g", go[0]))

    def cam(row):
        x = gpu_prepare(torch.from_numpy(np.ascontiguousarray(imgs[[row]])), train=False).requires_grad_(True)
        out = model(x)
        model.zero_grad(); out[0, 1].backward()
        w = grads["g"].mean((2, 3), keepdim=True)
        c = F.relu((w * feats["a"]).sum(1, keepdim=True))
        c = F.interpolate(c, size=96, mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        return c / (c.max() + 1e-8)

    groups = [("tumeur détectée", tp), ("tumeur ratée", fn), ("fausse alerte", fp)]
    fig, axes = plt.subplots(len(groups) * 2, 4, figsize=(9, 2.35 * len(groups) * 2))
    for g, (title, sel) in enumerate(groups):
        for k in range(4):
            a0, a1 = axes[2 * g, k], axes[2 * g + 1, k]
            for ax in (a0, a1):
                ax.axis("off")
            if k >= len(sel):
                continue
            r = sel.index[k]
            im = imgs[r]
            a0.imshow(im); a0.set_title(f"{title}\np = {sel.prob.iloc[k]:.2f}", fontsize=8)
            a1.imshow(im); a1.imshow(cam(r), cmap="jet", alpha=0.45, vmin=0, vmax=1)
    fig.suptitle(f"Grad-CAM du ResNet18 sur l'hôpital 2 (seuil {thr:.2f} appris sur val_id)", fontsize=10)
    fig.tight_layout()
    out = Path(f"reports/figures/gradcam_c2_{a.tag}.png")
    fig.savefig(out, dpi=130)
    print(f"Sortie : {out}")


if __name__ == "__main__":
    main()
