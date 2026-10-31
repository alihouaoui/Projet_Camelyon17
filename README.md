# Projet 1 — Détection de métastases (CAMELYON17)

Chaîne d'analyse : patchs histologiques → classification tumeur / non-tumeur → (plus tard) agrégation lame → stade pN.

## Données

- **Patchs** : Camelyon17-WILDS (Koh et al., 2021), 455 954 patchs RGB 96×96 issus de 50 lames annotées, 5 hôpitaux.
  Récupérés depuis le miroir Hugging Face [`wltjr1007/Camelyon17-WILDS`](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS),
  car le serveur officiel CodaLab a un certificat HTTPS expiré (30/10/2026). Effectifs vérifiés identiques à WILDS.
- **Lames complètes** (semaine 2) : CAMELYON17 sur AWS Open Data, `s3://camelyon-dataset/CAMELYON17/`.

## Découpage (sans fuite)

| Split | Contenu | Usage |
|---|---|---|
| `train` | centres 0, 3, 4, 80 % des patientes | entraînement |
| `val_id` | centres 0, 3, 4, 20 % des patientes (autres patientes) | choix du modèle et du seuil |
| `val_ood` | centre 1 | généralisation à un hôpital inconnu |
| `test_ood` | centre 2 | test final, touché une seule fois |

Le split `id_val` officiel de WILDS reprend des patchs **des mêmes lames** que l'entraînement : il est écarté.
`src/data.py:check_no_leakage` refuse tout découpage où une patiente apparaît dans deux splits.

## Installation et commandes

```bash
pip install -r requirements.txt
python -m scripts.fetch_sample                  # local : ~40 000 patchs tirés des 21 fichiers (~0,9 Go, ~1 h)
python -m scripts.baseline_classic              # référence couleur + texture
python -m scripts.train_cnn --config configs/resnet18_smoke.yaml   # test rapide local
```

## Jeu complet sur Google Colab

`notebooks/02_colab.ipynb` enchaîne tout : vérification du GPU, Google Drive, clonage du dépôt, téléchargement
des 21 parquets (10,7 Go), conversion, référence classique, ResNet18, petit CNN, synthèse.

- **Mémoire** : Colab gratuit n'a que 12,7 Go de RAM. `scripts/build_memmap.py` décode les images une fois dans
  `images.npy` (~12,6 Go sur le disque Colab), lues ensuite à la demande ; un seul parquet est en mémoire à la fois.
- **Coupures de session** : les résultats vont sur Drive (`MyDrive/Camelyon17/outputs`), `last.pt` est écrit après
  chaque epoch, et `resume: true` reprend où l'entraînement s'était arrêté.
- **Test honnête** : chaque epoch n'évalue que `val_id` et `val_ood` ; le centre 2 n'est évalué qu'une fois, à la fin.
- **Accès au dépôt privé** : jeton GitHub en lecture seule, rangé dans les *Secrets* de Colab sous `GITHUB_TOKEN`.

Chaque run écrit dans `<out>/<run>/` : `config.json`, `history.csv`, `metrics.csv` (par split et par centre,
au seuil « sensibilité ≥ 95 % sur val_id » et au seuil 0,5), `predictions.csv` (réutilisé pour les cartes de chaleur).

## Organisation

- `src/` : code réutilisable (données, modèles, augmentations, métriques, caractéristiques).
- `scripts/` : points d'entrée.
- `configs/` : un YAML par expérience.
- `notebooks/01_exploration.ipynb` : analyse exploratoire ; `notebooks/02_colab.ipynb` : jeu complet sur Colab
  (les deux sont générés par `scripts/build_*_notebook.py`).
- `JOURNAL.md` : décisions et résultats semaine par semaine.

## Sources et outils

WILDS (Koh et al., 2021) ; CAMELYON17 (Bándi et al., 2019) ; augmentation HED (Tellez et al., 2019) ;
PyTorch, torchvision, scikit-learn, scikit-image. Du code a été écrit avec l'aide d'un assistant IA
(Claude) ; chaque membre en a relu et compris chaque partie.
