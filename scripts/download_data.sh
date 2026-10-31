#!/usr/bin/env bash
# Télécharge les parquets Camelyon17-WILDS depuis le miroir Hugging Face.
#   bash scripts/download_data.sh all      # 21 fichiers, ~10,7 Go (Colab)
#   bash scripts/download_data.sh sample   # 1 fichier par split, ~1,5 Go (local)
# Le serveur officiel WILDS (CodaLab) a un certificat HTTPS expiré au 30/10/2026.
set -euo pipefail
MODE="${1:-sample}"
DEST="${2:-data/raw}"
BASE=https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/resolve/main/data
mkdir -p "$DEST"
if [ "$MODE" = "all" ]; then
  FILES=$(for i in $(seq -f "%05g" 0 13); do echo "train-$i-of-00014"; done
          for i in $(seq -f "%05g" 0 2); do echo "validation-$i-of-00003"; done
          for i in $(seq -f "%05g" 0 3); do echo "test-$i-of-00004"; done)
else
  FILES="train-00000-of-00014 validation-00000-of-00003 test-00000-of-00004"
fi
for f in $FILES; do
  [ -s "$DEST/$f.parquet" ] && { echo "déjà là : $f"; continue; }
  echo "-> $f"; curl -fL --retry 3 -o "$DEST/$f.parquet" "$BASE/$f.parquet"
done
