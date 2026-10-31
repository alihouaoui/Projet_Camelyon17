#!/usr/bin/env bash
# Suit la chaîne lancée sur Colab (scripts/colab_pipeline.sh) depuis le Mac, via la CLI Colab officielle.
#   bash scripts/colab_watch.sh [session] [intervalle_s]
# Toutes les N secondes : affiche les dernières étapes et rapatrie outputs/ dans outputs_colab/
# (best.pt et last.pt compris, pour relancer avec scripts/colab_launch.sh si la session est coupée).
# S'arrête quand la chaîne est terminée ou en erreur, après un dernier rapatriement (modèles compris).
set -uo pipefail
S=${1:-camelyon}
EVERY=${2:-300}
C=${COLAB:-$HOME/.venvs/colab-cli/bin/colab}
cd "$(dirname "$0")/.."
mkdir -p outputs_colab

remote() { printf '%s\n' "$1" | "$C" exec -s "$S" --timeout 300 < /dev/stdin 2>&1; }

sync() {  # $1 = "light" (best.pt et last.pt inclus, pour pouvoir reprendre ailleurs) ou "full" (tout)
  local excl="--exclude=epoch*.pt"; [ "$1" = full ] && excl="--exclude=nothing"
  remote "import subprocess; subprocess.run('mkdir -p /content/outputs && tar czf /content/outputs.tgz $excl -C /content outputs', shell=True)" > /dev/null
  "$C" download -s "$S" /content/outputs.tgz outputs_colab/outputs.tgz < /dev/null > /dev/null 2>&1 \
    && tar xzf outputs_colab/outputs.tgz -C outputs_colab && rm -f outputs_colab/outputs.tgz
}

fails=0
while true; do
  log=$(remote 'import re
print("@@LOG@@")
s = open("/content/pipeline.log").read()
keep = [l for l in re.split(r"[\r\n]", s) if l.startswith(("===", "epoch=", "  epoch", "Terminé", "Caract", "Sorties", "Reprise"))
        or "Traceback" in l or "Error" in l or "patchs" in l]
print("\n".join(keep[-12:]))')
  echo "----- $(date '+%H:%M:%S')"
  if ! echo "$log" | grep -q "@@LOG@@"; then  # coupure réseau Mac <-> Colab : on réessaie
    fails=$((fails + 1)); echo "connexion à Colab impossible ($fails/6)"
    if [ "$fails" -ge 6 ]; then echo "ARRÊT : Colab injoignable depuis 30 min"; exit 3; fi
    sleep "$EVERY"; continue
  fi
  fails=0; log=$(echo "$log" | awk 'f; /@@LOG@@/{f=1}'); echo "$log"
  sync light
  if echo "$log" | grep -q "=== .* terminé"; then sync full; echo "FIN : chaîne terminée"; exit 0; fi
  if echo "$log" | grep -qE "Traceback|Error"; then sync light; echo "ARRÊT : erreur sur Colab"; exit 1; fi
  if ! "$C" sessions < /dev/null 2>&1 | grep -q "^\[$S\]"; then echo "ARRÊT : session Colab perdue"; exit 2; fi
  sleep "$EVERY"
done
