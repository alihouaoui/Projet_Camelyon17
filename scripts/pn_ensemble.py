"""Pistes 1 + 2 après l'ABMIL : ensemble de graines (confirmatoire) et combinaisons avec la forêt (exploratoire).

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python -m scripts.pn_ensemble --repeats 10 --workers 6

Plan figé avant lancement, après critique par 5 relecteurs :
- Piste 1 (confirmatoire) : ABMIL binaire de l'hybride entraîné avec 3 graines par pli (graine 1000*rep + pli
  + k*100000 ; k = 0 redonne l'entraînement de scripts/pn_abmil.py), moyenne des probabilités, seuil 0,5.
  Comparaison primaire fixée d'avance : hybride 3 graines - hybride 1 graine (k = 0), mêmes plis.
- Piste 2 (EXPLORATOIRE, idée née après les résultats, IC non confirmatoires) : réduire les macro ratées.
  « OU » : macro si la forêt dit macro, sinon hybride ; « ET » : macro si la forêt dit macro et l'ABMIL dit
  atteinte, sinon hybride. Critère regardé : macro ratées, sous contrainte de ne pas dégrader la sur-stadification.
- Relecture humaine : patientes dont le stade diffère entre l'hybride 3 graines et la forêt = signalées ; on compte
  la charge de relecture et les erreurs graves qui restent parmi les patientes non signalées.

Les entraînements (rep, pli, graine) tournent en parallèle (processus, 2 fils chacun) et écrivent leurs probabilités
dans <out>/probs/ : relancer reprend là où on s'est arrêté. L'agrégation est faite ensuite, en série.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.pn_abmil import clinical, fast_kappa, load_bags, predict, train
from scripts.pn_stage import PN, kappa, prepare, stage_patients

G = {}  # données de chaque processus, lues en mmap depuis le cache disque (fork se bloque sur macOS avec torch)


def init_worker(cache: str):
    torch.set_num_threads(2)
    for k in ("X", "off", "p_tile", "y_bin"):
        G[k] = np.load(Path(cache) / f"{k}.npy", mmap_mode="r" if k == "X" else None)


def job(task):
    rep, f, k, tr, te, path = task
    y = G["y_bin"][tr]
    model, mu, sd = train(G["X"], G["off"], G["p_tile"], tr, y, 2, 1000 * rep + f + k * 100000)
    pb, _ = predict(model, mu, sd, G["X"], G["off"], te)
    np.save(path, np.stack([te, pb[:, 1]]))
    return task[:3]


def hybrid(pos, rule):
    return np.where(pos, np.where(rule == "macro", "macro", "micro"), np.where(rule == "itc", "itc", "negative"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default="outputs/wsi")
    ap.add_argument("--probe", default="outputs/probe_phikon_local/probe.npz")
    ap.add_argument("--cohort", default="configs/pn_cohort.csv")
    ap.add_argument("--out", default="outputs/pn_ensemble")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "probs").mkdir(parents=True, exist_ok=True)

    C = prepare(args.wsi, args.probe, args.cohort)
    slides, pats = C.slides, C.pats
    folds = {rep: list(C.folds(rep)) for rep in range(args.repeats)}
    tasks = []
    for rep in range(args.repeats):
        for f, (p_tr, p_te) in enumerate(folds[rep]):
            tr = slides.index[slides.pid.isin(p_tr)].to_numpy()
            te = slides.index[slides.pid.isin(p_te)].to_numpy()
            for k in range(args.seeds):
                path = out / "probs" / f"r{rep}_f{f}_k{k}.npy"
                if not path.exists():
                    tasks.append((rep, f, k, tr, te, path))
    print(f"{len(tasks)} entraînements à faire", flush=True)
    if tasks:
        cache = out / "cache"
        if not (cache / "y_bin.npy").exists():
            cache.mkdir(exist_ok=True)
            X, off, p_tile = load_bags(slides, args.wsi, dict(np.load(args.probe)))
            np.save(cache / "X.npy", X)
            np.save(cache / "off.npy", off)
            np.save(cache / "p_tile.npy", p_tile)
            np.save(cache / "y_bin.npy", slides.slide_label.isin(["micro", "macro"]).to_numpy().astype(int))
            del X
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(str(cache),)) as pool:
            for n, (rep, f, k) in enumerate(pool.imap_unordered(job, tasks), 1):
                print(f"Entraînement {n}/{len(tasks)} (rép {rep}, pli {f}, graine {k}) — "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)

    # agrégation
    methods = ["règle de taille", "forêt aléatoire", "ABMIL hybride 1 graine", "ABMIL hybride 3 graines",
               "explo. OU forêt macro", "explo. ET forêt macro"]
    rows, pat_rows = [], []
    for rep in range(args.repeats):
        sp = {m: pd.Series(index=slides.index, dtype=object) for m in methods}
        for f, (p_tr, p_te) in enumerate(folds[rep]):
            te = slides.index[slides.pid.isin(p_te)].to_numpy()
            t, m = C.best_rule(p_tr)
            rule = C.rule_slides(te, t, m)
            rf = C.fit_forest(p_tr).predict(slides.loc[te, C.feat_cols])
            P = [np.load(out / "probs" / f"r{rep}_f{f}_k{k}.npy") for k in range(args.seeds)]
            assert all((p[0].astype(int) == te).all() for p in P)
            p1, p3 = P[0][1], np.mean([p[1] for p in P], 0)
            h3 = hybrid(p3 >= 0.5, rule)
            sp["règle de taille"][te] = rule
            sp["forêt aléatoire"][te] = rf
            sp["ABMIL hybride 1 graine"][te] = hybrid(p1 >= 0.5, rule)
            sp["ABMIL hybride 3 graines"][te] = h3
            sp["explo. OU forêt macro"][te] = np.where(rf == "macro", "macro", h3)
            sp["explo. ET forêt macro"][te] = np.where((rf == "macro") & (p3 >= 0.5), "macro", h3)
        preds = {m: stage_patients(slides.assign(c=s), "c").loc[pats.index] for m, s in sp.items()}
        flag = preds["ABMIL hybride 3 graines"] != preds["forêt aléatoire"]
        for name, s in sp.items():
            pred = preds[name]
            r = {"methode": name, "repetition": rep, "kappa_pN": kappa(pats.pn_stage, pred),
                 "accord_pN": float((pred == pats.pn_stage).mean()),
                 "accord_lame": float((s == slides.slide_label).mean()), **clinical(pats.pn_stage, pred)}
            if name == "ABMIL hybride 3 graines":  # relecture humaine des désaccords avec la forêt
                r["signalees"] = int(flag.sum())
                rest = clinical(pats.pn_stage[~flag], pred[~flag])
                r["metastatiques_ratees_non_signalees"] = rest["metastatiques_ratees"]
                r["macro_ratees_non_signalees"] = rest["macro_ratees"]
            rows.append(r)
            pat_rows.append(pd.DataFrame({"repetition": rep, "methode": name, "pid": pats.index, "pred": pred.values}))

    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv.csv", index=False)
    pp = pd.concat(pat_rows).merge(pats.pn_stage.rename("verite"), left_on="pid", right_index=True)
    pp.to_csv(out / "patients.csv", index=False)
    res = cv.groupby("methode", sort=False).mean(numeric_only=True).drop(columns="repetition")
    res.insert(1, "kappa_min", cv.groupby("methode", sort=False).kappa_pN.min())
    res.insert(2, "kappa_max", cv.groupby("methode", sort=False).kappa_pN.max())
    res.to_csv(out / "resume.csv")

    pids = pats.index.to_numpy()
    yi = pats.pn_stage.map(PN.index).to_numpy()
    Pm = {(mth, r): g.set_index("pid").loc[pids, "pred"].map(PN.index).to_numpy()
          for (mth, r), g in pp.groupby(["methode", "repetition"])}
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(pids), len(pids)) for _ in range(args.boot)]
    reps = range(args.repeats)
    comps = [("confirmatoire (primaire)", "ABMIL hybride 3 graines", "ABMIL hybride 1 graine"),
             ("confirmatoire", "ABMIL hybride 3 graines", "règle de taille"),
             ("confirmatoire", "ABMIL hybride 3 graines", "forêt aléatoire"),
             ("exploratoire", "explo. OU forêt macro", "ABMIL hybride 3 graines"),
             ("exploratoire", "explo. ET forêt macro", "ABMIL hybride 3 graines")]
    ecarts = []
    for statut, a, b in comps:
        point = np.mean([fast_kappa(yi, Pm[a, r]) - fast_kappa(yi, Pm[b, r]) for r in reps])
        boot = [np.mean([fast_kappa(yi[d], Pm[a, r][d]) - fast_kappa(yi[d], Pm[b, r][d]) for r in reps])
                for d in draws]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ecarts.append({"statut": statut, "comparaison": f"{a} - {b}", "delta_kappa": point, "ic_bas": lo, "ic_haut": hi})
    ecarts = pd.DataFrame(ecarts)
    ecarts.to_csv(out / "ecarts.csv", index=False)

    # contrôle de reproductibilité : la graine k = 0 doit redonner l'hybride de scripts/pn_abmil.py
    prev = Path("outputs/pn_abmil/cv.csv")
    if prev.exists():
        old = pd.read_csv(prev).query("methode == 'ABMIL hybride'").set_index("repetition").kappa_pN
        new = cv.query("methode == 'ABMIL hybride 1 graine'").set_index("repetition").kappa_pN
        common = old.index.intersection(new.index)
        print(f"Reproductibilité graine 0 : écart max de kappa {np.abs(old[common] - new[common]).max():.4f}")

    with pd.option_context("display.width", 250, "display.precision", 3, "display.max_columns", 20):
        print(f"\n{C.n_folds} plis x {args.repeats} répétitions, {len(pats)} patientes")
        print(res)
        print("\nÉcarts de kappa (IC 95 % bootstrap des patientes) :")
        print(ecarts.to_string(index=False))
    print(f"\nSorties : {out}")


if __name__ == "__main__":
    main()
