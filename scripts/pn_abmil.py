"""ABMIL (attention multi-instances) pour l'étape 2 « classe du ganglion », comparé à la règle de taille et à la forêt.

    python -m scripts.pn_abmil --repeats 10

Plan figé avant tout lancement, après critique par 5 relecteurs :
- l'attention fait une moyenne pondérée des tuiles : elle ne « voit » ni la taille ni la forme d'une tache, donc ne peut
  pas trancher micro (0,2-2 mm) contre macro (> 2 mm). D'où :
  * PRINCIPAL — hybride : ABMIL binaire (métastase micro/macro ou non ; ITC compté négatif) décide si la lame est
    atteinte ; la taille vient de la carte de la sonde (règle t, m apprise sur les plis d'entraînement) :
    atteinte -> macro si la règle dit macro, sinon micro ; non atteinte -> itc si la règle dit itc, sinon négative ;
  * SECONDAIRE — ABMIL 3 classes direct (négative+itc / micro / macro), pour montrer ce qu'il ne sait pas faire.
- entrée : caractéristiques Phikon seules (ni coordonnées ni probabilité de la sonde) ; standardisation apprise sur
  les lames d'entraînement ; hyperparamètres fixés à l'avance, 20 époques, pas d'arrêt anticipé, une graine ;
- entraînement : toutes les tuiles de probabilité sonde >= 0,3 + tirage aléatoire jusqu'à 8 000 tuiles par lame
  (pour ne pas perdre une micro-métastase de 3 tuiles) ; inférence sur toutes les tuiles ;
- mêmes plis que scripts/pn_stage.py (règle et forêt recalculées dans la même boucle), kappa pondéré quadratique ;
  écart de kappa apparié par répétition, IC 95 % par bootstrap des patientes ; métriques cliniques asymétriques.

Sorties dans <out>/ : cv.csv, resume.csv, ecarts.csv, patients.csv (toutes répétitions), attention.npz (dernière
répétition, chaque lame vue en test).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.pn_stage import PN, kappa, prepare, stage_patients

METASTATIC = ["pN1mi", "pN1", "pN2"]
CLASSES3 = ["negative", "micro", "macro"]
MAX_TILES, P_KEEP, EPOCHS = 8000, 0.3, 20


class ABMIL(nn.Module):
    """Attention à porte (Ilse et al., 2018), version réduite (~110 k paramètres)."""

    def __init__(self, n_out: int, d_in: int = 768, d: int = 128, d_att: int = 64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(0.25))
        self.att_v = nn.Linear(d, d_att)
        self.att_u = nn.Linear(d, d_att)
        self.att_w = nn.Linear(d_att, 1)
        self.cls = nn.Linear(d, n_out)

    def forward(self, x):
        h = self.fc(x)
        a = torch.softmax(self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1), 0)
        return self.cls(a @ h), a


def load_bags(slides: pd.DataFrame, wsi: str, probe: dict):
    """Caractéristiques de toutes les lames dans un seul tableau fp16 (+ bornes) et probabilité sonde par tuile."""
    feats, off = [], [0]
    for s in slides.slide:
        f = np.load(Path(wsi) / s.replace(".tif", ".npz"))["feats"]
        feats.append(f)
        off.append(off[-1] + len(f))
    X = np.concatenate(feats)
    p = np.concatenate([1 / (1 + np.exp(-(((X[i:i + 200_000].astype(np.float32) - probe["mu"]) / probe["sd"])
                                          @ probe["w"] + probe["b"]))) for i in range(0, len(X), 200_000)])
    return X, np.array(off), p


def train(X, off, p_tile, idx, y, n_out, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sample = np.concatenate([np.arange(off[i], off[i + 1])[rng.random(off[i + 1] - off[i]) < 0.1] for i in idx])
    mu = X[sample].astype(np.float32).mean(0)
    sd = X[sample].astype(np.float32).std(0) + 1e-6
    w = torch.tensor(len(y) / (n_out * np.bincount(y, minlength=n_out).clip(1)), dtype=torch.float32)
    model = ABMIL(n_out)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(EPOCHS):
        for k in rng.permutation(len(idx)):
            i = idx[k]
            tiles = np.arange(off[i], off[i + 1])
            if len(tiles) > MAX_TILES:
                keep = tiles[p_tile[tiles] >= P_KEEP]
                if len(keep) >= MAX_TILES:
                    tiles = rng.choice(keep, MAX_TILES, replace=False)
                else:
                    rest = np.setdiff1d(tiles, keep, assume_unique=True)
                    tiles = np.concatenate([keep, rng.choice(rest, MAX_TILES - len(keep), replace=False)])
            x = torch.from_numpy((X[np.sort(tiles)].astype(np.float32) - mu) / sd)
            logits, _ = model(x)
            loss = lossf(logits[None], torch.tensor([y[k]]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd


@torch.no_grad()
def predict(model, mu, sd, X, off, idx):
    probs, atts = [], []
    for i in idx:
        logits, a = model(torch.from_numpy((X[off[i]:off[i + 1]].astype(np.float32) - mu) / sd))
        probs.append(torch.softmax(logits, -1).numpy())
        atts.append(a.numpy().astype(np.float16))
    return np.array(probs), atts


def clinical(y: pd.Series, pred: pd.Series) -> dict:
    met = y.isin(METASTATIC)
    return {"metastatiques_ratees": int((met & ~pred.isin(METASTATIC)).sum()),
            "macro_ratees": int((y.isin(["pN1", "pN2"]) & ~pred.isin(["pN1", "pN2"])).sum()),
            "sur_stadification": int((~met & pred.isin(METASTATIC)).sum())}


def fast_kappa(yi: np.ndarray, pi: np.ndarray, k: int = len(PN)) -> float:
    o = np.bincount(yi * k + pi, minlength=k * k).reshape(k, k).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    wq = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2).astype(float)
    return 1 - (wq * o).sum() / (wq * e).sum() if (wq * e).sum() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default="outputs/wsi")
    ap.add_argument("--probe", default="outputs/probe_phikon_local/probe.npz")
    ap.add_argument("--cohort", default="configs/pn_cohort.csv")
    ap.add_argument("--out", default="outputs/pn_abmil")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    C = prepare(args.wsi, args.probe, args.cohort)
    slides, pats = C.slides, C.pats
    X, off, p_tile = load_bags(slides, args.wsi, dict(np.load(args.probe)))
    print(f"{len(X):,} tuiles chargées ({X.nbytes / 1e9:.1f} Go)", flush=True)
    y_bin = slides.slide_label.isin(["micro", "macro"]).to_numpy().astype(int)
    y_3 = slides.slide_label.replace({"itc": "negative"}).map(CLASSES3.index).to_numpy()

    rows, pat_rows, attention = [], [], {}
    for rep in range(args.repeats):
        t0 = time.time()
        sp = {m: pd.Series(index=slides.index, dtype=object) for m in ("règle de taille", "forêt aléatoire",
                                                                        "ABMIL hybride", "ABMIL 3 classes")}
        for f, (p_tr, p_te) in enumerate(C.folds(rep)):
            tr = slides.index[slides.pid.isin(p_tr)].to_numpy()
            te = slides.index[slides.pid.isin(p_te)].to_numpy()
            t, m = C.best_rule(p_tr)
            rule = C.rule_slides(te, t, m)
            sp["règle de taille"][te] = rule
            sp["forêt aléatoire"][te] = C.fit_forest(p_tr).predict(slides.loc[te, C.feat_cols])
            seed = 1000 * rep + f
            mb, mu, sd = train(X, off, p_tile, tr, y_bin[tr], 2, seed)
            pb, att = predict(mb, mu, sd, X, off, te)
            pos = pb[:, 1] >= 0.5
            sp["ABMIL hybride"][te] = np.where(pos, np.where(rule == "macro", "macro", "micro"),
                                               np.where(rule == "itc", "itc", "negative"))
            m3, mu3, sd3 = train(X, off, p_tile, tr, y_3[tr], 3, seed)
            p3, _ = predict(m3, mu3, sd3, X, off, te)
            sp["ABMIL 3 classes"][te] = np.array(CLASSES3)[p3.argmax(1)]
            if rep == args.repeats - 1:
                attention.update({slides.slide[i]: a for i, a in zip(te, att)})
        for name, s in sp.items():
            pred = stage_patients(slides.assign(c=s), "c").loc[pats.index]
            rows.append({"methode": name, "repetition": rep, "kappa_pN": kappa(pats.pn_stage, pred),
                         "accord_pN": float((pred == pats.pn_stage).mean()),
                         "accord_lame": float((s == slides.slide_label).mean()), **clinical(pats.pn_stage, pred)})
            pat_rows.append(pd.DataFrame({"repetition": rep, "methode": name, "pid": pats.index, "pred": pred.values}))
        cur = pd.DataFrame(rows)
        cur = cur[cur.repetition == rep].set_index("methode").kappa_pN.round(3).to_dict()
        print(f"Répétition {rep + 1}/{args.repeats} ({time.time() - t0:.0f} s) : {cur}", flush=True)

    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv.csv", index=False)
    pp = pd.concat(pat_rows).merge(pats.pn_stage.rename("verite"), left_on="pid", right_index=True)
    pp.to_csv(out / "patients.csv", index=False)
    np.savez_compressed(out / "attention.npz", **{k.replace(".tif", ""): v for k, v in attention.items()})
    res = cv.groupby("methode").agg(kappa_moyen=("kappa_pN", "mean"), kappa_min=("kappa_pN", "min"),
                                    kappa_max=("kappa_pN", "max"), accord_pN=("accord_pN", "mean"),
                                    accord_lame=("accord_lame", "mean"),
                                    metastatiques_ratees=("metastatiques_ratees", "mean"),
                                    macro_ratees=("macro_ratees", "mean"),
                                    sur_stadification=("sur_stadification", "mean"))
    res.to_csv(out / "resume.csv")

    # écart de kappa apparié (mêmes plis), IC 95 % par bootstrap des patientes (mêmes tirages pour tous)
    pids = pats.index.to_numpy()
    yi = pats.pn_stage.map(PN.index).to_numpy()
    P = {(mth, r): g.set_index("pid").loc[pids, "pred"].map(PN.index).to_numpy()
         for (mth, r), g in pp.groupby(["methode", "repetition"])}
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(pids), len(pids)) for _ in range(args.boot)]
    reps = range(args.repeats)
    ecarts = []
    for a in ("ABMIL hybride", "ABMIL 3 classes"):
        for b in ("règle de taille", "forêt aléatoire"):
            point = np.mean([fast_kappa(yi, P[a, r]) - fast_kappa(yi, P[b, r]) for r in reps])
            boot = [np.mean([fast_kappa(yi[d], P[a, r][d]) - fast_kappa(yi[d], P[b, r][d]) for r in reps])
                    for d in draws]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ecarts.append({"comparaison": f"{a} - {b}", "delta_kappa": point, "ic_bas": lo, "ic_haut": hi})
    ecarts = pd.DataFrame(ecarts)
    ecarts.to_csv(out / "ecarts.csv", index=False)

    with pd.option_context("display.width", 200, "display.precision", 3):
        print(f"\n{C.n_folds} plis x {args.repeats} répétitions, {len(pats)} patientes")
        print(res)
        print("\nÉcarts de kappa (IC 95 % bootstrap des patientes) :")
        print(ecarts.to_string(index=False))
        last = pp[pp.repetition == args.repeats - 1]
        for name in ("ABMIL hybride", "ABMIL 3 classes"):
            g = last[last.methode == name]
            print(f"\n{name} — pN, dernière répétition (lignes = vérité) :")
            print(pd.crosstab(pd.Categorical(g.verite, PN), pd.Categorical(g.pred, PN), dropna=False))
    print(f"\nSorties : {out}")


if __name__ == "__main__":
    main()
