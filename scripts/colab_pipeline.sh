#!/usr/bin/env bash
# Chaîne complète sur une machine Colab (lancée en tâche de fond, journal dans /content/pipeline.log).
#   nohup bash scripts/colab_pipeline.sh > /content/pipeline.log 2>&1 &
# Chaque étape déjà faite est sautée : on peut relancer le script après une coupure.
# STAGES choisit les étapes (défaut « classic cnn ») ; « fm » = modèles de fondation figés + sonde linéaire ;
# « wsi » = caractéristiques Phikon des lames entières de la cohorte pN (lues à distance, sans les patchs WILDS).
set -euo pipefail
cd /content/Projet_Camelyon17
OUT=${OUT:-/content/outputs}
STAGES=${STAGES:-classic cnn}
want() { [[ " $STAGES " == *" $1 "* ]]; }
DATA=/content/data/full
mkdir -p "$OUT"
step() { echo; echo "=== $(date '+%H:%M:%S') $* ==="; }

step "dépendances"
pip install -q scikit-image pyarrow pyyaml joblib

if { want classic || want cnn || want fm; } && [ ! -f "$DATA/metadata.parquet" ]; then
  step "téléchargement des 21 parquets (10,7 Go)"
  bash scripts/download_data.sh all /content/raw
  step "conversion en images.npy"
  python -m scripts.build_memmap --raw /content/raw --out "$DATA"
  rm -rf /content/raw
fi
df -h /content | tail -1

if want classic && ! ls "$OUT"/classic_*/metrics.csv >/dev/null 2>&1; then
  step "référence classique"
  python -m scripts.baseline_classic --data "$DATA" --max-train 30000 --max-eval 20000 --out "$OUT"
fi

for run in resnet18 small_cnn; do
  if want cnn && [ ! -f "$OUT/${run}_full/metrics.csv" ]; then
    step "$run"
    python -m scripts.train_cnn --config "configs/$run.yaml" --data "$DATA" --run "${run}_full" --out "$OUT"
  fi
done
# modèles de fondation : du plus rapide au plus lent ; Midnight (ViT-g, ~28 img/s sur T4) sur 100 000 patchs
if want fm; then
  pip install -q -U "transformers>=4.56"
  for spec in phikon:0 dinov2_base:0 phikon_v2:0 midnight:100000; do
    m=${spec%%:*}; n=${spec##*:}
    if [ ! -f "$OUT/probe_$m/metrics.csv" ]; then
      step "caractéristiques $m"
      python -m scripts.extract_features --model "$m" --data "$DATA" --out "$OUT/features" --max-rows "$n" \
        --chunk $([ "$m" = midnight ] && echo 5000 || echo 20000) --batch-size $([ "$m" = midnight ] && echo 64 || echo 256)
      step "sonde $m"
      python -m scripts.linear_probe --features "$OUT/features/$m" --out "$OUT"
    fi
  done
fi
if want wsi; then
  pip install -q -U "transformers>=4.56" tifffile imagecodecs fsspec aiohttp
  step "lames entières (cohorte pN)"
  # Colab part de la fin de la cohorte : le Mac peut traiter le début en parallèle
  python -m scripts.wsi_features --cohort configs/pn_cohort.csv --out "$OUT/wsi" --reverse
fi
step "terminé"
