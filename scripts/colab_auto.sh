#!/usr/bin/env bash
# Relance automatique d'une étape Colab tant qu'elle n'est pas terminée :
#   STAGES=wsi bash scripts/colab_auto.sh [session] [heures_max]
# Toutes les 20 min : demande un GPU T4 (refusé quand le quota gratuit est épuisé) ; dès qu'il est
# accordé, lance la chaîne (scripts/colab_launch.sh) puis la suit (scripts/colab_watch.sh).
# Si Google coupe la session, on recommence ; la reprise saute ce qui est déjà fait.
set -uo pipefail
S=${1:-camelyon}
HOURS=${2:-20}
C=${COLAB:-$HOME/.venvs/colab-cli/bin/colab}
cd "$(dirname "$0")/.."
end=$(( $(date +%s) + HOURS * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  if "$C" sessions < /dev/null 2>&1 | grep -q "^\[$S\]" || "$C" new -s "$S" --gpu T4 < /dev/null > /dev/null 2>&1; then
    echo "$(date '+%H:%M') GPU obtenu, lancement"
    bash scripts/colab_launch.sh "$S" && bash scripts/colab_watch.sh "$S" 240
    code=$?
    if [ "$code" -eq 0 ]; then echo "$(date '+%H:%M') FIN"; "$C" stop -s "$S" < /dev/null > /dev/null 2>&1; exit 0; fi
    echo "$(date '+%H:%M') arrêt (code $code), nouvel essai dans 20 min"
    "$C" stop -s "$S" < /dev/null > /dev/null 2>&1
  else
    echo "$(date '+%H:%M') pas de GPU disponible, nouvel essai dans 20 min"
  fi
  sleep 1200
done
echo "ABANDON après $HOURS h"
exit 4
