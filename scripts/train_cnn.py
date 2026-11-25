"""Entraîne un classifieur de patchs (CNN) et l'évalue par split et par centre.

Exemples :
    # test rapide en local sur le sous-échantillon
    python -m scripts.train_cnn --config configs/resnet18_smoke.yaml
    # entraînement complet (Colab, GPU)
    python -m scripts.train_cnn --config configs/resnet18.yaml

Sorties dans <out>/<run>/ : config.json, history.csv, metrics.csv, predictions.csv, best.pt, last.pt

Sur Colab, `--out` pointe vers Google Drive et `--resume true` reprend depuis last.pt
après une coupure de session. Le centre 2 (test) n'est évalué qu'une fois, à la fin.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import PatchDataset, assign_splits, load_data, subsample  # noqa: E402
from src.metrics import report, threshold_for_sensitivity  # noqa: E402
from src.models import build_model  # noqa: E402
from src.transforms import eval_transform, gpu_prepare, train_transform  # noqa: E402

DEFAULTS = dict(
    data="data/processed/subset.parquet", out="outputs", run=None, model="resnet18", pretrained=True,
    epochs=5, batch_size=128, lr=1e-4, weight_decay=1e-4, balance="none", hed=False, size=96,
    max_train=None, max_eval=None, epoch_eval_max=20000, target_sensitivity=0.95, num_workers=4, seed=0,
    val_frac=0.2, resume=False,
)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    for k, v in DEFAULTS.items():
        t = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool) else (type(v) if v is not None else str)
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t, default=None)
    ns = ap.parse_args()
    cfg = dict(DEFAULTS)
    if ns.config:
        cfg.update(yaml.safe_load(open(ns.config)))
    cfg.update({k: v for k, v in vars(ns).items() if v is not None and k != "config"})
    for k in ("max_train", "max_eval", "epoch_eval_max"):
        cfg[k] = int(cfg[k]) or None if cfg[k] is not None else None  # 0 = pas de limite
    cfg["run"] = cfg["run"] or f"{cfg['model']}_{time.strftime('%Y%m%d-%H%M%S')}"
    return cfg


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(dev):
    """Précision mixte sur GPU CUDA (Colab) ; sans effet ailleurs."""
    return torch.autocast("cuda") if dev.type == "cuda" else contextlib.nullcontext()


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def predict(model, loader, dev, gpu_aug=False):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(dev, non_blocking=True)
        if gpu_aug:
            x = gpu_prepare(x, train=False)
        with autocast(dev):
            logits = model(x)
        out.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    cfg = get_args()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out = Path(cfg["out"]) / cfg["run"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    dev = device()
    print(f"Run {cfg['run']} sur {dev}")

    df, images_path = load_data(cfg["data"])
    df = assign_splits(df, val_frac=cfg["val_frac"], seed=cfg["seed"])
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    tr = subsample(df[df.split == "train"], cfg["max_train"], cfg["seed"])
    evals = {s: subsample(df[df.split == s], cfg["max_eval"], cfg["seed"]) for s in ("val_id", "val_ood", "test_ood")}
    evals = {s: d for s, d in evals.items() if len(d)}
    # suivi par epoch : validations seulement, sous-échantillonnées pour gagner du temps
    epoch_evals = {s: subsample(evals[s], cfg["epoch_eval_max"], cfg["seed"]) for s in ("val_id", "val_ood") if s in evals}

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0,
              pin_memory=dev.type == "cuda")
    # sans HED ni redimensionnement, les augmentations se font par lot sur le GPU (bien plus rapide)
    # (la HED est aussi faite par lot sur le GPU : src.transforms.hed_jitter_batch, vérifiée identique à skimage)
    gpu_aug = cfg["size"] == 96
    train_ds = PatchDataset(tr, None if gpu_aug else train_transform(cfg["hed"], cfg["size"]), images_path)
    counts = np.bincount(train_ds.labels, minlength=2)
    class_w = torch.tensor(len(train_ds.labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32)
    if cfg["balance"] == "sampler":
        sampler = WeightedRandomSampler(class_w[train_ds.labels].double(), len(train_ds), replacement=True)
        train_dl = DataLoader(train_ds, sampler=sampler, **kw)
    else:
        train_dl = DataLoader(train_ds, shuffle=True, **kw)
    eval_tf = None if gpu_aug else eval_transform(cfg["size"])
    make_dl = lambda d: DataLoader(PatchDataset(d, eval_tf, images_path), shuffle=False, **kw)  # noqa: E731
    epoch_dls = {s: make_dl(d) for s, d in epoch_evals.items()}

    model = build_model(cfg["model"], cfg["pretrained"]).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=class_w.to(dev) if cfg["balance"] == "weighted_loss" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * len(train_dl))

    amp_ok = dev.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=amp_ok) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp_ok))

    history, best_auc, start = [], -1.0, 1
    if cfg["resume"] and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)  # fichier produit par ce script
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, best_auc, start = ck["history"], ck["best_auc"], ck["epoch"] + 1
        print(f"Reprise après l'epoch {ck['epoch']}")

    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if gpu_aug:
                x = gpu_prepare(x, train=True, hed=cfg["hed"])
            with autocast(dev):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if step % 500 == 0:
                print(f"  epoch {epoch} pas {step}/{len(train_dl)} perte {np.mean(losses[-500:]):.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "time_s": time.time() - t0}
        for s, dl in epoch_dls.items():
            row[f"{s}_auc"] = float(safe_auc(epoch_evals[s].label, predict(model, dl, dev, gpu_aug)))
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        # sélection du modèle sur la validation interne uniquement (le centre 2 reste intouché)
        if row.get("val_id_auc", -row["train_loss"]) > best_auc:
            best_auc = row.get("val_id_auc", -row["train_loss"])
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(model.state_dict(), out / f"epoch{epoch}.pt")  # pour comparer après coup le choix sur val_id ou val_ood
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": history, "best_auc": float(best_auc)},
                   out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    preds = []
    for s, d in evals.items():
        d = d[["image_id", "split", "center", "patient", "node", "slide_id", "x_coord", "y_coord", "label"]].copy()
        d["prob"] = predict(model, make_dl(evals[s]), dev, gpu_aug)
        preds.append(d)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(out / "predictions.csv", index=False)

    # seuil fixé sur val_id pour atteindre la sensibilité cible, puis appliqué tel quel aux autres splits
    ref = preds[preds.split == "val_id"]
    thr = threshold_for_sensitivity(ref.label.values, ref.prob.values, cfg["target_sensitivity"])
    metrics = pd.concat([report(preds, thr).assign(rule=f"sens>={cfg['target_sensitivity']}"),
                         report(preds, 0.5).assign(rule="thr=0.5")], ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(metrics[metrics.center == "all"])
    print(f"Sorties : {out}")


if __name__ == "__main__":
    main()
