#!/usr/bin/env bash
# Lance (ou relance après une coupure) la chaîne complète sur Colab depuis le Mac, via la CLI Colab officielle.
#   bash scripts/colab_launch.sh [session]
# 1. crée une session GPU T4 si elle n'existe pas ;
# 2. envoie le code, et les résultats déjà rapatriés dans outputs_colab/ (pour que `resume` reprenne) ;
# 3. démarre scripts/colab_pipeline.sh en tâche de fond sur Colab.
# Suivi ensuite avec scripts/colab_watch.sh.
set -euo pipefail
S=${1:-camelyon}
C=${COLAB:-$HOME/.venvs/colab-cli/bin/colab}
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)

remote() { printf '%s\n' "$1" | "$C" exec -s "$S" --timeout 300; }

if ! "$C" sessions < /dev/null 2>&1 | grep -q "^\[$S\]"; then
  echo "Création de la session $S (GPU T4)..."
  "$C" new -s "$S" --gpu T4 < /dev/null
fi

COPYFILE_DISABLE=1 tar czf "$TMP/code.tgz" --exclude=data --exclude=outputs --exclude=outputs_colab \
  --exclude=__pycache__ --exclude=.git --exclude=reports --exclude='*.ipynb' .
"$C" upload -s "$S" "$TMP/code.tgz" /content/code.tgz < /dev/null
remote 'import subprocess; subprocess.run("mkdir -p /content/Projet_Camelyon17 && tar xzf /content/code.tgz -C /content/Projet_Camelyon17", shell=True, check=True)'

if [ -d outputs_colab/outputs ]; then
  echo "Envoi des résultats déjà obtenus (reprise)..."
  COPYFILE_DISABLE=1 tar czf "$TMP/outputs.tgz" -C outputs_colab outputs
  "$C" upload -s "$S" "$TMP/outputs.tgz" /content/outputs_prev.tgz < /dev/null
  remote 'import subprocess; subprocess.run("tar xzf /content/outputs_prev.tgz -C /content", shell=True, check=True)'
fi
rm -rf "$TMP"

remote 'import subprocess
subprocess.Popen("nohup bash scripts/colab_pipeline.sh >> /content/pipeline.log 2>&1 &", shell=True,
                 cwd="/content/Projet_Camelyon17", start_new_session=True)
print("chaîne démarrée")'
