"""Figure de la comparaison principale du rapport : CNN contre Phikon (sensibilité et AUC sur C1 et C2).

    python -m scripts.fig_cnn_vs_phikon      (après scripts.compare)
Sortie : reports/figures/cnn_vs_phikon.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODELES = ["ResNet18 + HED", "Phikon (figé)"]   # meilleur CNN (choisi sur l'AUC de l'hôpital 1) contre Phikon
COUL = ["#C44E52", "#2E7D32"]

r = pd.read_csv("reports/comparaison_modeles.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
for ax, (met, titre) in zip(axes, [("sens", "Sensibilité au seuil appris ailleurs"), ("auc", "AUC")]):
    x = np.arange(2)
    w = 0.32
    for k, (m, c) in enumerate(zip(MODELES, COUL)):
        d = r[r.modele == m].set_index("centre").loc[["C1", "C2"]]
        v, lo, hi = d[met].values, d[f"{met}_ic_bas"].values, d[f"{met}_ic_haut"].values
        pos = x + (k - 0.5) * w
        ax.bar(pos, v, w, color=c, label=m.replace("ResNet18 + HED", "Meilleur CNN : ResNet18 + HED").replace(" (figé)", " figé + sonde"))
        ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.set_xticks(x, ["Hôpital 1 (inconnu)", "Hôpital 2 (test)"])
    ax.set_ylim(0.25 if met == "sens" else 0.7, 1.0)
    ax.set_title(titre)
    if met == "sens":
        ax.axhline(0.95, ls="--", c="#C44E52", lw=1)
        ax.text(-0.45, 0.957, "cible 95 %", color="#C44E52", ha="left", fontsize=8)
axes[1].legend(fontsize=8, loc="lower left")
fig.suptitle("Meilleur CNN contre Phikon sur les hôpitaux jamais vus (IC 95 % par bootstrap des patientes)", fontsize=11)
fig.tight_layout()
fig.savefig("reports/figures/cnn_vs_phikon.png", dpi=150)
print("Sortie : reports/figures/cnn_vs_phikon.png")
