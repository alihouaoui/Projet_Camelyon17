#!/usr/bin/env bash
# Chaîne complète sur une machine Colab (lancée en tâche de fond, journal dans /content/pipeline.log).
#   nohup bash scripts/colab_pipeline.sh > /content/pipeline.log 2>&1 &
# Chaque étape déjà faite est sautée : on peut relancer le script après une coupure.
set -euo pipefail
cd /content/Projet_Camelyon17
OUT=${OUT:-/content/outputs}
DATA=/content/data/full
mkdir -p "$OUT"
step() { echo; echo "=== $(date '+%H:%M:%S') $* ==="; }

step "dépendances"
pip install -q scikit-image pyarrow pyyaml joblib

if [ ! -f "$DATA/metadata.parquet" ]; then
  step "téléchargement des 21 parquets (10,7 Go)"
  bash scripts/download_data.sh all /content/raw
  step "conversion en images.npy"
  python -m scripts.build_memmap --raw /content/raw --out "$DATA"
  rm -rf /content/raw
fi
df -h /content | tail -1

if ! ls "$OUT"/classic_*/metrics.csv >/dev/null 2>&1; then
  step "référence classique"
  python -m scripts.baseline_classic --data "$DATA" --max-train 30000 --max-eval 20000 --out "$OUT"
fi

for run in resnet18 small_cnn; do
  if [ ! -f "$OUT/${run}_full/metrics.csv" ]; then
    step "$run"
    python -m scripts.train_cnn --config "configs/$run.yaml" --data "$DATA" --run "${run}_full" --out "$OUT"
  fi
done
step "terminé"
