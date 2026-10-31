"""Génère notebooks/02_colab.ipynb : entraînement sur le jeu complet, Colab + Google Drive + GitHub."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 02 — Entraînement sur le jeu complet (Colab)

**Avant de lancer**
1. *Exécution → Modifier le type d'exécution → GPU T4.*
2. Une seule fois : dans la barre de gauche, icône 🔑 *Secrets*, ajouter `GITHUB_TOKEN`
   (jeton GitHub en lecture seule sur le dépôt) et activer l'accès pour ce notebook.

**Où vont les choses**
| Quoi | Où | Pourquoi |
|---|---|---|
| Code | GitHub, cloné dans `/content/Projet_Camelyon17` | une seule version partagée par le groupe |
| Données (455 954 patchs) | disque Colab `/content/data/full` | rapide ; effacé à la fin de la session |
| Résultats et modèles | Google Drive `MyDrive/Camelyon17/outputs` | survivent aux coupures ; `resume` reprend où on s'était arrêté |""")

md("## 1. Vérifier le GPU")
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "PAS DE GPU : Exécution > Modifier le type d'exécution > T4"
import torch; print('CUDA :', torch.cuda.is_available())""")

md("## 2. Google Drive (résultats)")
code("""from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/Camelyon17'
OUT = f'{DRIVE}/outputs'
!mkdir -p "$OUT"
print('Résultats dans', OUT)""")

md("## 3. Code depuis GitHub")
code("""from google.colab import userdata
import os, subprocess
REPO = 'alihouaoui/Projet_Camelyon17'
token = userdata.get('GITHUB_TOKEN')
if not os.path.exists('/content/Projet_Camelyon17'):
    subprocess.run(['git', 'clone', '-q', f'https://{token}@github.com/{REPO}.git', '/content/Projet_Camelyon17'], check=True)
else:
    subprocess.run(['git', '-C', '/content/Projet_Camelyon17', 'pull', '-q'], check=True)
%cd /content/Projet_Camelyon17
!git log --oneline -1""")

md("""## 4. Données complètes

Téléchargement des 21 parquets (10,7 Go, ~5 à 10 min), puis conversion en images décodées sur
disque (`images.npy`, ~12,6 Go, ~15 min). On lit ensuite les images à la demande : la RAM
(12,7 Go) ne sature pas.

`CACHE_DRIVE = True` garde aussi les parquets sur Drive (10,7 Go) pour éviter de les retélécharger
aux séances suivantes : seulement si ton Drive a la place.""")
code("""CACHE_DRIVE = False
RAW_DRIVE = f'{DRIVE}/raw'
if os.path.exists('/content/data/full/metadata.parquet'):
    print('Déjà prêt dans cette session')
else:
    if os.path.isdir(RAW_DRIVE) and len(os.listdir(RAW_DRIVE)) == 21:
        !mkdir -p /content/raw && cp "$RAW_DRIVE"/*.parquet /content/raw/
    else:
        !bash scripts/download_data.sh all /content/raw
        if CACHE_DRIVE:
            !mkdir -p "$RAW_DRIVE" && cp /content/raw/*.parquet "$RAW_DRIVE"/
    !python -m scripts.build_memmap --raw /content/raw --out /content/data/full
    !rm -rf /content/raw
!df -h /content | tail -1; free -g | head -2""")

md("""## 5. Référence classique (couleur + texture)
30 000 patchs d'entraînement suffisent pour ce type de modèle ; évaluation sur 20 000 patchs par split.""")
code("""!python -m scripts.baseline_classic --data /content/data/full --max-train 30000 --max-eval 20000 --out "$OUT\"""")

md("""## 6. ResNet18 pré-entraîné
5 epochs. Si la session coupe : relancer les cellules 1 à 4, puis celle-ci — `resume: true` reprend
depuis `last.pt` sur Drive. Le centre 2 (test) n'est évalué qu'à la fin.""")
code("""!python -m scripts.train_cnn --config configs/resnet18.yaml --run resnet18_full --out "$OUT\"""")

md("## 7. Petit CNN entraîné de zéro")
code("""!python -m scripts.train_cnn --config configs/small_cnn.yaml --run small_cnn_full --out "$OUT\"""")

md("## 8. Synthèse")
code("""import pandas as pd, glob
rows = []
for f in sorted(glob.glob(f'{OUT}/*/metrics.csv')):
    m = pd.read_csv(f); m = m[m.center.astype(str) == 'all']
    m['run'] = f.split('/')[-2] + ('_' + m.model if 'model' in m else '')
    rows.append(m)
res = pd.concat(rows)
cols = ['run', 'rule', 'split', 'auc_roc', 'sensitivity', 'specificity', 'fn', 'fp']
display(res[res.rule.str.startswith('sens')][cols].pivot_table(index='run', columns='split',
        values=['auc_roc', 'sensitivity']).round(3))""")

md("**Fin de séance** : *Exécution → Déconnecter et supprimer l'environnement*, pour ne pas garder le GPU inutilement.")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_colab.ipynb"
nbf.write(nb, out)
print(out)
