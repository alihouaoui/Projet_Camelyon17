"""Sonde linéaire (régression logistique) sur des caractéristiques figées.

    python -m scripts.linear_probe --features outputs/features/phikon --out outputs

Protocole (corrige la faiblesse de val_id : 5 patientes, tumeurs venant presque toutes de 2) :
- « dev » = centres d'entraînement 0, 3, 4 (train + val_id) ;
- régularisation choisie par validation croisée groupée par patiente sur dev (5 plis) ;
- seuil de sensibilité 95 % fixé sur les prédictions hors-pli de dev, puis appliqué tel quel ;
- modèle final réentraîné sur tout dev, évalué sur C1 (val_ood) et C2 (test_ood).
Le seuil réglé sur C1 et le seuil 0,5 sont rapportés à côté. C2 n'intervient dans aucun choix.

Sorties dans <out>/probe_<modèle>/ : cv.csv, predictions.csv, metrics.csv (même format que train_cnn), probe.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import assign_splits  # noqa: E402
from src.metrics import report, threshold_for_sensitivity  # noqa: E402
from scripts.train_cnn import device  # noqa: E402


def load_features(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    index = pd.read_parquet(path / "index.parquet")
    files = sorted(path.glob("feats_[0-9][0-9][0-9][0-9].npy"))
    X = np.concatenate([np.load(f) for f in files]).astype(np.float32)
    if len(X) != len(index):
        raise ValueError(f"Extraction incomplète : {len(X)} caractéristiques pour {len(index)} patchs")
    index["patient_id"] = index["patient"].astype(int)
    return index, X


def fit(X, y, wd, dev, epochs=15, batch=4096, seed=0):
    """Régression logistique avec pénalité L2 `wd` (perte moyenne + wd/2 ||w||²), par Adam."""
    torch.manual_seed(seed)
    X, y = torch.as_tensor(X, device=dev), torch.as_tensor(y, device=dev, dtype=torch.float32)
    lin = nn.Linear(X.shape[1], 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    steps = epochs * ((len(X) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), device=dev).split(batch):
            loss = nn.functional.binary_cross_entropy_with_logits(lin(X[idx]).squeeze(1), y[idx])
            loss = loss + wd / 2 * lin.weight.pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return lin


@torch.no_grad()
def predict(lin, X, dev):
    return torch.sigmoid(lin(torch.as_tensor(X, device=dev)).squeeze(1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--run")
    ap.add_argument("--wd", default="1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feats = Path(args.features)
    out = Path(args.out) / (args.run or f"probe_{feats.name}")
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    df, X = load_features(feats)
    df = assign_splits(df, seed=args.seed)  # même découpage par patiente que les CNN
    dev_mask = df.split.isin(["train", "val_id"]).to_numpy()
    print(f"Sonde {feats.name} : {X.shape[0]} patchs, dim {X.shape[1]}, dev {dev_mask.sum()}", flush=True)

    # standardisation avec les statistiques de dev uniquement
    mu, sd = X[dev_mask].mean(0), X[dev_mask].std(0) + 1e-6
    X = (X - mu) / sd
    Xd, yd, gd = X[dev_mask], df.label.to_numpy()[dev_mask], df.patient_id.to_numpy()[dev_mask]

    folds = list(StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed).split(Xd, yd, gd))
    rows, oof_by_wd = [], {}
    for wd in [float(w) for w in args.wd.split(",")]:
        oof = np.zeros(len(yd))
        for k, (tr, va) in enumerate(folds):
            oof[va] = predict(fit(Xd[tr], yd[tr], wd, dev, seed=args.seed), Xd[va], dev)
        fold_auc = [roc_auc_score(yd[va], oof[va]) for _, va in folds if len(np.unique(yd[va])) == 2]
        rows.append({"wd": wd, "auc_oof": roc_auc_score(yd, oof), "auc_fold_mean": float(np.mean(fold_auc)),
                     "auc_fold_min": float(np.min(fold_auc))})
        oof_by_wd[wd] = oof
        print("Sonde " + " | ".join(f"{k}={v:.4g}" for k, v in rows[-1].items()), flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv.csv", index=False)
    best_wd = float(cv.loc[cv.auc_fold_mean.idxmax(), "wd"])

    lin = fit(Xd, yd, best_wd, dev, seed=args.seed)
    # sonde finale, réutilisable sur d'autres tuiles (lames entières) : p = sigmoïde(((x - mu) / sd) @ w + b)
    np.savez(out / "probe.npz", mu=mu, sd=sd, w=lin.weight.detach().cpu().numpy()[0], b=lin.bias.item())
    keep = ["image_id", "split", "center", "patient", "node", "slide_id", "x_coord", "y_coord", "label"]
    preds = df.loc[dev_mask, keep].assign(split="dev_oof", prob=oof_by_wd[best_wd])
    ext = df.loc[~dev_mask, keep].assign(prob=predict(lin, X[~dev_mask], dev))
    preds = pd.concat([preds, ext], ignore_index=True)
    preds.to_csv(out / "predictions.csv", index=False)

    t = args.target_sensitivity
    oof = preds[preds.split == "dev_oof"]
    c1 = preds[preds.split == "val_ood"]
    thr_dev = threshold_for_sensitivity(oof.label.values, oof.prob.values, t)
    thr_c1 = threshold_for_sensitivity(c1.label.values, c1.prob.values, t) if len(c1) else np.nan
    metrics = pd.concat([report(preds, thr_dev).assign(rule=f"sens>={t} (dev hors-pli)"),
                         report(preds, thr_c1).assign(rule=f"sens>={t} (C1)"),
                         report(preds, 0.5).assign(rule="thr=0.5")], ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    json.dump({**vars(args), "best_wd": best_wd, "thr_dev": thr_dev, "thr_c1": thr_c1},
              open(out / "config.json", "w"), indent=2)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(metrics[metrics.center == "all"][["rule", "split", "n", "auc_roc", "threshold", "sensitivity",
                                                "specificity", "fn"]])
    print(f"Sorties : {out}", flush=True)


if __name__ == "__main__":
    main()
