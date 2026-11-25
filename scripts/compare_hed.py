"""Effet de l'augmentation HED-light : comparaison appariée avec les runs de référence.

    python -m scripts.compare_hed outputs_colab/outputs

Plan fixé avant de voir les résultats (relu par 5 critiques) :
- mêmes réglages, graine et découpage que la référence ; seule différence : hed=true ;
- critère principal : C1 (val_ood) ; C2 regardé une fois ;
- écarts HED - référence avec IC 95 % par bootstrap APPARIÉ des patientes (mêmes tirages) ;
- diagnostic seuil / séparabilité : sensibilité au seuil val_id, et spécificité au seuil « oracle »
  qui donnerait 95 % de sensibilité sur le centre lui-même (non utilisable en pratique) ;
- niveau patiente : patientes avec tumeur dont aucun patch tumoral n'est détecté.
Résultat exploratoire : une seule graine, HED faible (±5 %), val_id de 5 patientes.

Sortie : reports/hed_comparaison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PAIRS = {"ResNet18": ("resnet18_full", "resnet18_hed_full"), "Petit CNN": ("small_cnn_full", "small_cnn_hed_full")}
SPLITS = {"val_ood": "C1", "test_ood": "C2"}


def load(d: Path):
    m = pd.read_csv(d / "metrics.csv")
    thr = float(m[m.rule.str.startswith("sens>=")].threshold.iloc[0])
    return pd.read_csv(d / "predictions.csv"), thr


def stats(y, p, thr):
    pos, neg = np.sort(p[y == 1]), p[y == 0]
    oracle = pos[int(np.floor(0.05 * len(pos)))]
    return {"auc": roc_auc_score(y, p), "sens": (pos >= thr).mean(), "spec": (neg < thr).mean(),
            "spec_oracle": (neg < oracle).mean()}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs_colab/outputs")
    rng, B = np.random.default_rng(0), 1000
    rows = []
    for name, (ref, hed) in PAIRS.items():
        if not (root / hed / "metrics.csv").exists():
            print(f"{name} : run HED absent, sauté")
            continue
        (pr, tr), (ph, th) = load(root / ref), load(root / hed)
        for split, short in SPLITS.items():
            a = pr[pr.split == split].set_index("image_id")
            b = ph[ph.split == split].set_index("image_id").loc[a.index]   # mêmes patchs, même ordre
            pats = np.sort(a.patient.unique())
            idx = {k: np.where(a.patient.values == k)[0] for k in pats}
            y, p0, p1 = a.label.values, a.prob.values, b.prob.values
            s0, s1 = stats(y, p0, tr), stats(y, p1, th)
            boot = []
            for _ in range(B):
                sel = np.concatenate([idx[k] for k in rng.choice(pats, len(pats), replace=True)])
                if len(np.unique(y[sel])) < 2:
                    continue
                b0, b1 = stats(y[sel], p0[sel], tr), stats(y[sel], p1[sel], th)
                boot.append({k: b1[k] - b0[k] for k in b0})
            boot = pd.DataFrame(boot)
            # niveau patiente : patientes avec tumeur dont aucun patch tumoral n'est détecté
            tum = a[a.label == 1]
            miss0 = int((tum.assign(d=p0[a.label.values == 1] >= tr).groupby("patient").d.max() == 0).sum())
            miss1 = int((tum.assign(d=p1[a.label.values == 1] >= th).groupby("patient").d.max() == 0).sum())
            for k in s0:
                lo, hi = np.percentile(boot[k], [2.5, 97.5])
                rows.append({"modele": name, "centre": short, "metrique": k, "reference": s0[k], "hed": s1[k],
                             "ecart": s1[k] - s0[k], "ic_bas": lo, "ic_haut": hi})
            rows.append({"modele": name, "centre": short, "metrique": "patientes_tumeur_ratees",
                         "reference": miss0, "hed": miss1, "ecart": miss1 - miss0,
                         "ic_bas": np.nan, "ic_haut": tum.patient.nunique()})
            print(f"{name} {short} : seuils réf {tr:.3f} / HED {th:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv("reports/hed_comparaison.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(res)


if __name__ == "__main__":
    main()
