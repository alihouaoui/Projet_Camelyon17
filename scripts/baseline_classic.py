"""Référence classique : couleur + texture (LBP, GLCM) -> régression logistique et forêt aléatoire.

Mêmes splits, même règle de seuil et mêmes métriques que le CNN, pour une comparaison honnête.

    python -m scripts.baseline_classic --data data/processed/subset.parquet --max-train 20000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_cnn import subsample  # noqa: E402
from src.data import ImageSource, assign_splits, load_data  # noqa: E402
from src.features import extract  # noqa: E402
from src.metrics import report, threshold_for_sensitivity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/subset.parquet", help="parquet(s) ou dossier memmap")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, images_path = load_data(args.data)
    df = assign_splits(df, seed=args.seed)
    parts = {"train": subsample(df[df.split == "train"], args.max_train, args.seed)}
    parts.update({s: subsample(df[df.split == s], args.max_eval, args.seed) for s in ("val_id", "val_ood", "test_ood")})
    parts = {s: d for s, d in parts.items() if len(d)}
    t0 = time.time()
    X = {s: extract(ImageSource(d, images_path)) for s, d in parts.items()}
    print(f"Caractéristiques : {X['train'].shape[1]} par patch, {time.time() - t0:.0f} s")

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=args.seed),
    }
    out = Path(args.out) / f"classic_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for name, m in models.items():
        m.fit(X["train"], parts["train"].label.values)
        preds = pd.concat([parts[s][["image_id", "split", "center", "patient", "node", "slide_id", "label"]]
                           .assign(prob=m.predict_proba(X[s])[:, 1]) for s in parts if s != "train"], ignore_index=True)
        preds.to_csv(out / f"predictions_{name}.csv", index=False)
        ref = preds[preds.split == "val_id"]
        thr = threshold_for_sensitivity(ref.label.values, ref.prob.values, args.target_sensitivity)
        all_metrics.append(report(preds, thr).assign(model=name, rule=f"sens>={args.target_sensitivity}"))
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(metrics[metrics.center == "all"])
    print(f"Sorties : {out}")


if __name__ == "__main__":
    main()
