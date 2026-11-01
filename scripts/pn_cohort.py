"""Construit la cohorte « stade pN » : les patientes CAMELYON17 jamais vues par le détecteur de patchs.

    python -m scripts.pn_cohort --wilds-index outputs_colab/outputs/features/phikon/index.parquet

stages.csv (officiel, 100 patientes x 5 lames) donne le pN par patiente et l'étiquette de chaque lame
(negative / itc / micro / macro). Les 43 patientes présentes dans WILDS sont exclues : le détecteur
a été entraîné ou validé sur leurs patchs. Restent 57 patientes (285 lames).

Ordre de traitement « utilisable à tout moment » : hôpitaux hors domaine d'abord (1, 2), puis 0, 3, 4,
et à l'intérieur de chaque hôpital alternance des stades pN. Si le calcul s'arrête en route,
les premières patientes traitées couvrent déjà tous les hôpitaux et tous les stades.

Sortie : configs/pn_cohort.csv (une ligne par lame).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STAGES_URL = "https://camelyon-dataset.s3.amazonaws.com/CAMELYON17/stages.csv"
CENTER_ORDER = [1, 2, 0, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wilds-index", required=True, help="métadonnées des patchs WILDS (colonne patient)")
    ap.add_argument("--out", default="configs/pn_cohort.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    st = pd.read_csv(STAGES_URL)
    pat = st[st.patient.str.endswith(".zip")].assign(pid=lambda d: d.patient.str[8:11].astype(int))
    slides = st[st.patient.str.endswith(".tif")].assign(pid=lambda d: d.patient.str[8:11].astype(int))
    seen = set(pd.read_parquet(args.wilds_index, columns=["patient"]).patient.astype(int))
    pat = pat[~pat.pid.isin(seen)]

    order = []
    for c in CENTER_ORDER:
        p = pat[pat.center == c].sample(frac=1, random_state=args.seed)
        p = p.assign(k=p.groupby("stage").cumcount()).sort_values(["k", "stage"])  # alternance des stades
        order.append(p.assign(rank_in_center=np.arange(len(p))))
    order = pd.concat(order).sort_values(["rank_in_center"], kind="stable")
    order = order.assign(order=np.arange(len(order)))[["pid", "center", "stage", "order"]]
    order = order.rename(columns={"stage": "pn_stage"})
    out = slides.rename(columns={"patient": "slide", "stage": "slide_label"}).merge(order, on=["pid", "center"])
    out = out.sort_values(["order", "slide"])[["order", "pid", "center", "pn_stage", "slide", "slide_label"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{out.pid.nunique()} patientes, {len(out)} lames -> {args.out}")
    print(pd.crosstab(out.drop_duplicates("pid").center, out.drop_duplicates("pid").pn_stage))
    print(out.slide_label.value_counts().to_dict())


if __name__ == "__main__":
    main()
