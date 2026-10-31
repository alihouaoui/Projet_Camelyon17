"""Compare tous les modèles sur C1 et C2 avec des intervalles de confiance par patiente.

    python -m scripts.compare outputs_colab/outputs

Les patchs d'une même patiente sont très corrélés : un intervalle calculé au niveau patch serait
beaucoup trop étroit. On rééchantillonne donc les PATIENTES (bootstrap, mêmes tirages pour tous
les modèles), puis on recalcule l'AUC et la sensibilité / spécificité au seuil appris hors de C1/C2
(premier seuil « sens>= » du metrics.csv de chaque run : val_id pour les CNN, dev hors-pli pour les sondes).

Sorties : reports/comparaison_modeles.csv, reports/figures/sensibilite_c2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

NAMES = {"classic": "Couleur+texture (RF)", "small_cnn_full": "Petit CNN", "resnet18_full": "ResNet18 fine-tuné",
         "probe_dinov2_base": "DINOv2 générique (figé)", "probe_phikon": "Phikon (figé)",
         "probe_phikon_v2": "Phikon-v2 (figé)", "probe_midnight": "Midnight-12k (figé, 100k)"}
SPLITS = {"val_ood": "C1", "test_ood": "C2"}


def stats(y, p, thr):
    return {"auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "sens": (p[y == 1] >= thr).mean(), "spec": (p[y == 0] < thr).mean()}


def runs(root: Path):
    for d in sorted(root.iterdir()):
        if not (d / "predictions.csv").exists() or not (d / "metrics.csv").exists():
            continue
        m = pd.read_csv(d / "metrics.csv")
        sens_rules = [r for r in m.rule.unique() if r.startswith("sens>=")]
        thr = float(m[m.rule == sens_rules[0]].threshold.iloc[0])
        pred = pd.read_csv(d / "predictions.csv")
        if "prob" not in pred:  # référence classique : une colonne de probabilité par modèle
            continue
        key = "classic" if d.name.startswith("classic") else d.name
        yield NAMES.get(key, d.name), pred, thr, sens_rules[0]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs_colab/outputs")
    B, rng = 1000, np.random.default_rng(0)
    all_runs = list(runs(root))
    rows = []
    for split, short in SPLITS.items():
        patients = np.sort(pd.concat([p[p.split == split].patient for _, p, _, _ in all_runs]).unique())
        draws = [rng.choice(patients, len(patients), replace=True) for _ in range(B)]  # communs à tous les modèles
        for name, pred, thr, rule in all_runs:
            d = pred[pred.split == split]
            by_pat = {k: g for k, g in d.groupby("patient")}
            point = stats(d.label.values, d.prob.values, thr)
            boot = []
            for draw in draws:
                g = pd.concat([by_pat[k] for k in draw if k in by_pat])
                boot.append(stats(g.label.values, g.prob.values, thr))
            boot = pd.DataFrame(boot)
            row = {"modele": name, "centre": short, "n_patchs": len(d), "n_patientes": d.patient.nunique(),
                   "seuil": thr, "regle_seuil": rule}
            for k in ("auc", "sens", "spec"):
                lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
                row.update({k: point[k], f"{k}_ic_bas": lo, f"{k}_ic_haut": hi})
            rows.append(row)
    res = pd.DataFrame(rows)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    res.to_csv("reports/comparaison_modeles.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 3):
        print(res.drop(columns=["regle_seuil"]))

    c2 = res[res.centre == "C2"].sort_values("auc")
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(c2) + 1.8))
    yy = np.arange(len(c2))
    ax.barh(yy, c2.sens, color="#4C72B0", alpha=.85)
    ax.errorbar(c2.sens, yy, xerr=[c2.sens - c2.sens_ic_bas, c2.sens_ic_haut - c2.sens], fmt="none", ecolor="k", capsize=3)
    ax.axvline(0.95, ls="--", c="#C44E52", lw=1)
    ax.text(0.945, -0.6, "cible 95 % ", color="#C44E52", ha="right", va="center")
    ax.set_yticks(yy, [f"{n}\nAUC {a:.3f} · spécificité {sp:.2f}" for n, a, sp in zip(c2.modele, c2.auc, c2.spec)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Sensibilité sur l'hôpital de test (C2)\nseuil appris sans C1 ni C2 · IC 95 % par bootstrap des patientes")
    ax.set_title("Le seuil appris ailleurs se transfère-t-il à un nouvel hôpital ?")
    fig.tight_layout()
    fig.savefig("reports/figures/sensibilite_c2.png", dpi=150)
    print("Sorties : reports/comparaison_modeles.csv, reports/figures/sensibilite_c2.png")


if __name__ == "__main__":
    main()
