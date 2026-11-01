"""Stade pN par patiente à partir des cartes de probabilité des lames entières.

    python -m scripts.pn_stage --wsi outputs/wsi --probe outputs/probe_phikon_local/probe.npz

1. Carte par lame : sonde Phikon (régression logistique) appliquée à chaque tuile de tissu.
2. Lame -> négative / ITC / micro / macro par deux méthodes :
   A. règle de taille : seuil de probabilité t, composantes connexes d'au moins m tuiles,
      plus grand diamètre (Feret) en mm ; ITC < 0,2 mm <= micro <= 2 mm < macro (définitions TNM, non réglées) ;
   B. forêt aléatoire sur des résumés de la carte (probabilité max, nombre de tuiles suspectes, diamètres).
3. Patiente -> pN par les règles officielles du challenge (les ITC ne comptent pas comme ganglion positif).

Réglage honnête : validation croisée par patiente (stratifiée sur le stade pN, répétée 20 fois ; jusqu'à 5 plis,
limité par le stade le plus rare : 3 plis sur la cohorte complète, qui n'a que 3 patientes pN0(i+)).
Les paramètres (t, m) ou la forêt sont appris sur les plis d'entraînement et appliqués au pli restant ;
critère fixé à l'avance : kappa pondéré quadratique au niveau patiente (métrique officielle).
Aucune de ces patientes n'a servi à entraîner le détecteur de patchs.

Sorties dans <out>/ : slides.csv, patients.csv, cv.csv, resume.csv, figures cartes_*.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from skimage.measure import label, regionprops  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

SLIDE_CLASSES = ["negative", "itc", "micro", "macro"]
PN = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
MIN_TILES = [1, 2, 3, 5, 8]
ITC_MM, MACRO_MM = 0.2, 2.0


def heatmap(npz, probe) -> tuple[np.ndarray, float, np.ndarray]:
    d = np.load(npz)
    x = d["feats"].astype(np.float32)
    p = 1 / (1 + np.exp(-(((x - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]))) if len(x) else np.zeros(0)
    shape = d["shape"] // 96
    hm = np.full(shape, np.nan, np.float32)  # nan = pas de tissu
    if len(p):
        hm[d["grid"][:, 0], d["grid"][:, 1]] = p
    return hm, float(d["mpp"]) * 96 / 1000, d["thumb"]  # taille d'une tuile en mm


def largest_diameter(hm, t, m, tile_mm) -> float:
    """Plus grand diamètre (mm) parmi les composantes de >= m tuiles au-dessus du seuil t."""
    lab = label(np.nan_to_num(hm) >= t, connectivity=2)
    feret = [r.feret_diameter_max for r in regionprops(lab) if r.area >= m]
    return max(feret) * tile_mm if feret else 0.0


def size_class(diam_mm) -> np.ndarray:
    d = np.asarray(diam_mm)
    return np.select([d <= 0, d < ITC_MM, d <= MACRO_MM], ["negative", "itc", "micro"], "macro")


def pn_stage(slide_classes) -> str:
    c = list(slide_classes)
    pos = sum(k in ("micro", "macro") for k in c)
    if pos == 0:
        return "pN0(i+)" if "itc" in c else "pN0"
    if "macro" not in c:
        return "pN1mi"
    return "pN1" if pos <= 3 else "pN2"


def kappa(y, p) -> float:
    return cohen_kappa_score([PN.index(v) for v in y], [PN.index(v) for v in p], weights="quadratic",
                             labels=list(range(len(PN))))


def stage_patients(slides: pd.DataFrame, col: str) -> pd.Series:
    return slides.groupby("pid")[col].apply(pn_stage)


def summary_features(hm, tile_mm) -> dict:
    v = hm[~np.isnan(hm)]
    top = np.sort(v)[::-1] if len(v) else np.zeros(1)
    f = {"p_max": top[0], "p_top10": top[:10].mean(), "n_tissue": len(v)}
    for t in (0.5, 0.9, 0.99):
        f[f"n_{t}"] = int((v >= t).sum())
        f[f"diam_{t}"] = largest_diameter(hm, t, 1, tile_mm)
    return f


def prepare(wsi: str, probe_path: str, cohort_path: str) -> SimpleNamespace:
    """Cartes, diamètres et résumés de toutes les lames, plus les briques de la validation croisée
    (réutilisées par scripts/pn_abmil.py pour comparer sur exactement les mêmes plis)."""
    probe = dict(np.load(probe_path))
    cohort = pd.read_csv(cohort_path)
    done = cohort[cohort.slide.map(lambda s: (Path(wsi) / s.replace(".tif", ".npz")).exists())]
    complete = done.groupby("pid").slide.count().eq(5)  # une patiente n'est utilisable qu'avec ses 5 lames
    slides = done[done.pid.isin(complete[complete].index)].reset_index(drop=True)
    pats = slides.drop_duplicates("pid").set_index("pid")[["center", "pn_stage"]]
    print(f"{len(pats)} patientes complètes ({len(slides)} lames) sur {cohort.pid.nunique()}", flush=True)

    # diamètres pour toute la grille (t, m), et résumés pour la forêt
    diam, feats, maps = {}, [], {}
    for i, r in slides.iterrows():
        hm, tile_mm, thumb = heatmap(Path(wsi) / r.slide.replace(".tif", ".npz"), probe)
        maps[r.slide] = (hm, thumb)
        for t in THRESHOLDS:
            for m in MIN_TILES:
                diam[(i, t, m)] = largest_diameter(hm, t, m, tile_mm)
        feats.append(summary_features(hm, tile_mm))
    F = pd.DataFrame(feats)
    slides = pd.concat([slides, F], axis=1)

    def rule_slides(idx, t, m):
        return size_class([diam[(i, t, m)] for i in idx])

    def best_rule(pids):
        sl = slides[slides.pid.isin(pids)]
        y = pats.loc[pids, "pn_stage"]
        scores = {}
        for t in THRESHOLDS:
            for m in MIN_TILES:
                pred = stage_patients(sl.assign(c=rule_slides(sl.index, t, m)), "c").loc[pids]
                scores[(t, m)] = kappa(y, pred)
        return max(scores, key=scores.get)

    feat_cols = list(F.columns)

    def fit_forest(pids):
        sl = slides[slides.pid.isin(pids)]
        return RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(
            sl[feat_cols], sl.slide_label)

    # validation croisée répétée, par patiente, stratifiée sur le stade pN
    strat = pats.pn_stage.replace({"pN2": "pN1"})  # pN2 trop rare pour 5 plis
    n_folds = int(min(5, strat.value_counts().min()))

    def folds(rep):
        for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=rep).split(pats.index, strat):
            yield pats.index[tr], pats.index[te]

    return SimpleNamespace(slides=slides, pats=pats, maps=maps, feat_cols=feat_cols, n_folds=n_folds, folds=folds,
                           rule_slides=rule_slides, best_rule=best_rule, fit_forest=fit_forest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default="outputs/wsi")
    ap.add_argument("--probe", default="outputs/probe_phikon_local/probe.npz")
    ap.add_argument("--cohort", default="configs/pn_cohort.csv")
    ap.add_argument("--out", default="outputs/pn")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C = prepare(args.wsi, args.probe, args.cohort)
    slides, pats, maps, feat_cols, n_folds = C.slides, C.pats, C.maps, C.feat_cols, C.n_folds
    rule_slides, best_rule, fit_forest = C.rule_slides, C.best_rule, C.fit_forest

    rows, oof_last = [], None
    for rep in range(args.repeats):
        pred_rule = pd.Series(index=pats.index, dtype=object)
        pred_rf = pd.Series(index=pats.index, dtype=object)
        slide_rule = pd.Series(index=slides.index, dtype=object)
        slide_rf = pd.Series(index=slides.index, dtype=object)
        for p_tr, p_te in C.folds(rep):
            sl_te = slides[slides.pid.isin(p_te)]
            t, m = best_rule(p_tr)
            slide_rule[sl_te.index] = rule_slides(sl_te.index, t, m)
            slide_rf[sl_te.index] = fit_forest(p_tr).predict(sl_te[feat_cols])
        slides_rep = slides.assign(rule=slide_rule, rf=slide_rf)
        pred_rule = stage_patients(slides_rep, "rule").loc[pats.index]
        pred_rf = stage_patients(slides_rep, "rf").loc[pats.index]
        for name, pred, sp in (("règle de taille", pred_rule, slide_rule), ("forêt aléatoire", pred_rf, slide_rf)):
            rows.append({"methode": name, "repetition": rep, "kappa_pN": kappa(pats.pn_stage, pred),
                         "accord_pN": float((pred == pats.pn_stage).mean()),
                         "accord_lame": float((sp == slides.slide_label).mean())})
        oof_last = slides_rep, pred_rule, pred_rf
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv.csv", index=False)
    res = cv.groupby("methode").agg(kappa_moyen=("kappa_pN", "mean"), kappa_min=("kappa_pN", "min"),
                                    kappa_max=("kappa_pN", "max"), accord_pN=("accord_pN", "mean"),
                                    accord_lame=("accord_lame", "mean"))
    res.to_csv(out / "resume.csv")

    # détail d'une répétition (la dernière) : par hôpital, matrices de confusion
    slides_rep, pred_rule, pred_rf = oof_last
    slides_rep.drop(columns=feat_cols).to_csv(out / "slides.csv", index=False)
    pt = pats.assign(pred_regle=pred_rule, pred_foret=pred_rf)
    pt.to_csv(out / "patients.csv")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(f"\nValidation croisée {n_folds} plis x {args.repeats} répétitions, {len(pats)} patientes")
        print(res)
        for name, col, scol in (("Règle de taille", "pred_regle", "rule"), ("Forêt aléatoire", "pred_foret", "rf")):
            print(f"\n{name} — pN (lignes = vérité, colonnes = prédiction) :")
            print(pd.DataFrame(confusion_matrix(pt.pn_stage, pt[col], labels=PN), index=PN, columns=PN))
            print(f"{name} — lames :")
            print(pd.DataFrame(confusion_matrix(slides_rep.slide_label, slides_rep[scol], labels=SLIDE_CLASSES),
                               index=SLIDE_CLASSES, columns=SLIDE_CLASSES))
            by_c = pt.groupby("center").apply(lambda g: pd.Series(
                {"n": len(g), "kappa": kappa(g.pn_stage, g[col]) if g.pn_stage.nunique() > 1 else np.nan,
                 "accord": (g.pn_stage == g[col]).mean()}))
            print(f"{name} — par hôpital :\n{by_c}")

    # figures : cartes de chaleur pour une macro détectée, une métastase ratée, une fausse alerte
    sr = slides_rep
    cases = {"vrai positif (macro)": sr[(sr.slide_label == "macro") & (sr.rule == "macro")],
             "métastase ratée": sr[sr.slide_label.isin(["micro", "macro"]) & (sr.rule == "negative")],
             "fausse alerte": sr[(sr.slide_label == "negative") & sr.rule.isin(["micro", "macro"])]}
    picked = {k: v.sort_values("p_max", ascending=k != "vrai positif (macro)").iloc[0] for k, v in cases.items() if len(v)}
    if picked:
        fig, axes = plt.subplots(2, len(picked), figsize=(4.5 * len(picked), 9), squeeze=False)
        for j, (title, r) in enumerate(picked.items()):
            hm, thumb = maps[r.slide]
            ys, xs = np.where(~np.isnan(hm))
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            f = thumb.shape[0] / hm.shape[0]
            axes[0, j].imshow(thumb[int(y0 * f):int(y1 * f), int(x0 * f):int(x1 * f)])
            axes[0, j].set_title(f"{title}\n{r.slide.replace('.tif', '')} — vérité : {r.slide_label}", fontsize=10)
            im = axes[1, j].imshow(hm[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=1)
            axes[1, j].set_title(f"prédit : {r.rule} — p max {r.p_max:.2f}", fontsize=10)
            for a in axes[:, j]:
                a.axis("off")
        fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.6, label="probabilité de tumeur (tuile de 93 µm)")
        fig.savefig(out / "cartes_exemples.png", dpi=130, bbox_inches="tight")
    print(f"\nSorties : {out}")


if __name__ == "__main__":
    main()
