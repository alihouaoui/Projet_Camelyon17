# CAMELYON17 — Détection de métastases ganglionnaires et stade pN

**CNN classique contre modèle de fondation (Phikon) face au changement d'hôpital**

Équipe : Ali Houaoui · Benjamin Gardet · Mathys Vrillon · Rayan Abdoulkader Houmed

📄 Rapport : [`docs/rapport_final.pdf`](docs/rapport_final.pdf) · 📓 Journal de bord : [`JOURNAL.md`](JOURNAL.md)

---

## En bref

Sur des images de ganglions lymphatiques au microscope, nous voulons :

1. **Étape 1 — détecter le cancer** sur chaque patch de 96 × 96 pixels, avec un modèle qui reste fiable dans un **hôpital jamais vu** ;
2. **Étape 2 — donner le stade pN** de chaque patiente à partir de ses 5 lames de ganglions.

| Résultat principal | Hôpital 1 (inconnu) | Hôpital 2 (test) |
|---|---|---|
| Meilleur CNN (ResNet18 + augmentation de couleur HED) — cancers détectés | 88 % | **72 %** |
| **Phikon figé + régression logistique** — cancers détectés | **98 %** | **98 %** |

Seuil réglé sur les hôpitaux d'entraînement (sensibilité 95 %), puis appliqué tel quel. Pour le stade pN de 57 patientes
jamais vues, l'**ABMIL hybride** atteint un kappa de **0,89**, contre 0,80 pour une règle de taille et une forêt aléatoire.

---

## Sommaire

1. [Organisation du dépôt](#1-organisation-du-dépôt)
2. [Installation](#2-installation)
3. [Reproduire les résultats, étape par étape](#3-reproduire-les-résultats-étape-par-étape)
4. [Reproduire le rapport et les documents](#4-reproduire-le-rapport-et-les-documents)
5. [Où trouver chaque résultat du rapport](#5-où-trouver-chaque-résultat-du-rapport)
6. [Garanties de reproductibilité](#6-garanties-de-reproductibilité)
7. [Chronologie du projet](#7-chronologie-du-projet)
8. [Sources, outils et usage de l'IA](#8-sources-outils-et-usage-de-lia)

---

## 1. Organisation du dépôt

```
Projet_Camelyon17/
├── src/                     code réutilisable
│   ├── data.py              chargement, découpage par patiente, contrôle anti-fuite
│   ├── transforms.py        augmentations (rotations, miroirs, HED sur GPU)
│   ├── models.py            petit CNN, ResNet18
│   ├── metrics.py           seuil à sensibilité cible, métriques par hôpital
│   ├── features.py          caractéristiques couleur + texture (référence classique)
│   ├── foundation.py        modèles de fondation (Phikon, Phikon-v2, Midnight, DINOv2)
│   └── wsi.py               lecture à distance des lames entières, masque de tissu, tuiles
├── scripts/                 points d'entrée (un script = une étape)
├── configs/                 réglages des entraînements (YAML) et cohorte pN
├── notebooks/
│   ├── 01_exploration.ipynb         analyse exploratoire
│   ├── 02_colab.ipynb               chaîne complète sur Google Colab
│   └── 03_colab_cnn_explique.ipynb  notebook pédagogique autonome (données + CNN)
├── reports/                 tableaux de résultats (CSV) et figures du rapport
├── docs/                    rapport final et documents de suivi (LaTeX + PDF)
├── JOURNAL.md               décisions et résultats, séance par séance
└── requirements.txt
```

Les dossiers `data/`, `outputs/` et `outputs_colab/` (données et sorties lourdes) ne sont pas versionnés : ils sont recréés
par les commandes ci-dessous.

---

## 2. Installation

```bash
git clone https://github.com/alihouaoui/Projet_Camelyon17.git
cd Projet_Camelyon17
pip install -r requirements.txt
```

Python ≥ 3.10. Un GPU est conseillé pour le jeu complet (nous avons utilisé **Google Colab gratuit, GPU T4**) ;
tout le reste tourne sur un ordinateur portable.

---

## 3. Reproduire les résultats, étape par étape

Chaque commande se lance depuis la racine du dépôt. Les durées indiquées sont celles que nous avons mesurées.

### Étape 0 — Données

Le serveur officiel de WILDS ayant un certificat HTTPS expiré, les données viennent d'une **copie identique sur
Hugging Face** (`wltjr1007/Camelyon17-WILDS`, 21 fichiers Parquet, 10,7 Go, effectifs vérifiés).

**Option A — échantillon local** (mise au point, ~40 000 patchs, ~1 h) :
```bash
python -m scripts.fetch_sample --groups-per-file 20 --out data/processed/subset.parquet
```

**Option B — jeu complet** (455 954 patchs) :
```bash
bash scripts/download_data.sh all data/raw                          # 10,7 Go
python -m scripts.build_memmap --raw data/raw --out data/full       # images décodées sur disque (~12,6 Go)
```
Le format *memmap* permet de lire les images à la demande : le jeu complet ne tient pas en mémoire sur Colab gratuit.

> 💡 **Le plus simple pour découvrir le projet** : ouvrir `notebooks/03_colab_cnn_explique.ipynb` dans Google Colab,
> puis *Exécution → Tout exécuter*. Il télécharge lui-même un échantillon, explique les données, entraîne un CNN et
> calcule les métriques (environ 20 minutes).

### Étape 1 — Exploration des données
```bash
jupyter notebook notebooks/01_exploration.ipynb
```
Classes, couleur par hôpital (la couleur seule reconnaît l'hôpital dans 80 % des cas), dépendance spatiale des patchs.
Figures enregistrées dans `reports/figures/`.

### Étape 2 — Détection sur patchs (étape 1 du projet)

Le **découpage** est fait automatiquement par chaque script : hôpitaux 0, 3, 4 pour l'entraînement (80 % des patientes) et
la validation (20 %), hôpital 1 en validation hors domaine, hôpital 2 en test. Un contrôle refuse tout découpage où une
patiente apparaîtrait dans deux ensembles.

**Référence classique** (couleur + texture → régression logistique et forêt aléatoire) :
```bash
python -m scripts.baseline_classic --data data/full --max-train 30000 --max-eval 20000 --out outputs
```

**CNN** (petit CNN appris de zéro et ResNet18 pré-entraîné, avec et sans augmentation de couleur HED) :
```bash
python -m scripts.train_cnn --config configs/small_cnn.yaml --data data/full --run small_cnn_full --out outputs
python -m scripts.train_cnn --config configs/resnet18.yaml  --data data/full --run resnet18_full  --out outputs
python -m scripts.train_cnn --config configs/small_cnn.yaml --data data/full --run small_cnn_hed_full --out outputs --hed true
python -m scripts.train_cnn --config configs/resnet18.yaml  --data data/full --run resnet18_hed_full  --out outputs --hed true
```
≈ 250–270 s par époque sur GPU T4 (5 époques pour le petit CNN, 3 pour le ResNet18). Test rapide en local :
`python -m scripts.train_cnn --config configs/resnet18_smoke.yaml`.

**Modèles de fondation figés + sonde linéaire** (Phikon, Phikon-v2, Midnight, DINOv2) :
```bash
python -m scripts.extract_features --model phikon --data data/full --out outputs/features
python -m scripts.linear_probe --features outputs/features/phikon --out outputs
# idem avec --model phikon_v2, dinov2_base, et midnight --max-rows 100000
```
Phikon : ~20 min d'extraction sur T4, sonde en quelques minutes. Les modèles sont téléchargés depuis Hugging Face.

Chaque run écrit dans `outputs/<run>/` : `config.json`, `history.csv`, `metrics.csv` (par ensemble et par hôpital, au seuil
« sensibilité 95 % » et au seuil 0,5) et `predictions.csv`.

### Étape 3 — Comparaisons, figures et interprétabilité
```bash
python -m scripts.compare outputs            # tous les modèles, IC 95 % par bootstrap des patientes
python -m scripts.compare_hed outputs        # effet de l'augmentation HED (comparaison appariée)
python -m scripts.fig_cnn_vs_phikon          # figure principale : meilleur CNN contre Phikon
python -m scripts.gradcam --weights outputs/resnet18_full/best.pt --tag ref        # Grad-CAM
python -m scripts.gradcam --weights outputs/resnet18_hed_full/best.pt --tag hed
```
Sorties : `reports/comparaison_modeles.csv`, `reports/hed_comparaison.csv`, `reports/figures/*.png`.

### Étape 4 — Stade pN par patiente (étape 2 du projet)

**Cohorte** : les 57 patientes CAMELYON17 jamais vues par le détecteur (285 lames) :
```bash
python -m scripts.pn_cohort --wilds-index outputs/features/phikon/index.parquet   # → configs/pn_cohort.csv (déjà fourni)
```

**Cartes de chaleur** : lecture à distance des lames (bucket public AWS, requêtes HTTP Range, sans télécharger les 2 Go
par lame), tuiles de tissu 96 × 96, caractéristiques Phikon :
```bash
python -m scripts.wsi_features --cohort configs/pn_cohort.csv --out outputs/wsi      # ~1 lame/min sur T4
python -m scripts.linear_probe --features outputs/features/phikon --out outputs --run probe_phikon_local --wd 0.1
```

**Classe du ganglion puis stade pN** (validation croisée par patiente, mêmes plis pour toutes les méthodes) :
```bash
python -m scripts.pn_stage  --wsi outputs/wsi --probe outputs/probe_phikon_local/probe.npz   # règle de taille + forêt
python -m scripts.pn_abmil  --repeats 10                                                     # ABMIL hybride et 3 classes
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python -m scripts.pn_ensemble --repeats 10 --workers 6   # graines + combinaisons
```
CPU suffisant (~3 h 30 pour l'ABMIL). Résumés copiés dans `reports/pn_*.csv`.

### Tout lancer sur Google Colab depuis son ordinateur

Avec la CLI officielle `google-colab-cli`, la chaîne complète tourne sur Colab et les résultats sont rapatriés
automatiquement (reprise automatique après une coupure de session) :
```bash
STAGES="classic cnn" bash scripts/colab_launch.sh    # aussi : "fm" (fondation), "hed", "wsi"
bash scripts/colab_watch.sh                          # suivi + rapatriement dans outputs_colab/
```
Les scripts d'analyse acceptent alors `outputs_colab/outputs` comme dossier d'entrée.

---

## 4. Reproduire le rapport et les documents

Les documents sont écrits en LaTeX et utilisent les figures de `reports/figures/` :

```bash
cd docs
pdflatex rapport_final.tex && pdflatex rapport_final.tex     # 2 passes pour la table des matières
```

| Document | Contenu |
|---|---|
| `docs/rapport_final.pdf` | **le rapport rendu** (14 pages) |
| `docs/colab_cnn_explique.pdf` | explication détaillée du notebook Colab pédagogique |
| `docs/point_avancement.pdf` | point d'avancement de mi-projet |

Structure du rapport : 1. contexte et objectifs · 2. données · 3. méthodes · 4. résultats étape 1 (CNN contre Phikon) ·
5. résultats étape 2 (stade pN) · 6. changement de domaine · 7. interprétabilité · 8. éthique · 9. conclusion et limites.

---

## 5. Où trouver chaque résultat du rapport

| Dans le rapport | Script qui le produit | Fichier |
|---|---|---|
| Figures d'exploration (couleur, patchs, t-SNE, positions) | `notebooks/01_exploration.ipynb` | `reports/figures/couleur_par_centre.png`, `galerie.png`, `tsne_couleur.png`, `positions_lame.png` |
| Choix du meilleur CNN, CNN contre Phikon | `scripts/compare.py` | `reports/comparaison_modeles.csv` |
| Figure principale | `scripts/fig_cnn_vs_phikon.py` | `reports/figures/cnn_vs_phikon.png` |
| Sensibilité de tous les modèles | `scripts/compare.py` | `reports/figures/sensibilite_c2.png` |
| Effet de l'augmentation HED | `scripts/compare_hed.py` | `reports/hed_comparaison.csv` |
| Stade pN : règle et forêt | `scripts/pn_stage.py` | `reports/pn_resume.csv`, `pn_patients.csv` |
| Stade pN : ABMIL | `scripts/pn_abmil.py` | `reports/pn_abmil_resume.csv`, `pn_abmil_ecarts.csv` |
| Stade pN : graines et combinaisons | `scripts/pn_ensemble.py` | `reports/pn_ensemble_resume.csv`, `pn_ensemble_ecarts.csv` |
| Cartes de chaleur | `scripts/pn_stage.py` | `reports/figures/pn_cartes_exemples.png` |
| Grad-CAM | `scripts/gradcam.py` | `reports/figures/gradcam_c2_ref.png`, `gradcam_c2_hed.png` |

---

## 6. Garanties de reproductibilité

- **Graine fixée** (`seed: 0`) pour le découpage, les tirages et l'initialisation des réseaux. Sur GPU, quelques calculs
  restent non déterministes : les résultats peuvent varier à la troisième décimale.
- **Découpage par patiente** et **contrôle automatique anti-fuite** (`src/data.py:check_no_leakage`).
- **Réglages versionnés** : chaque entraînement est décrit par un fichier `configs/*.yaml`, recopié dans `config.json`.
- **Test honnête** : l'hôpital 2 n'est jamais évalué pendant l'entraînement ; l'époque, le seuil et le meilleur CNN sont
  choisis sans lui.
- **Incertitude rapportée** : intervalles de confiance à 95 % par bootstrap des patientes ; comparaisons appariées sur les
  mêmes tirages et les mêmes plis.
- **Plans d'expérience écrits avant lancement** ; les analyses imaginées après coup sont signalées comme exploratoires
  (voir `JOURNAL.md`).
- **Sauvegardes et reprise** : `last.pt` après chaque époque, reprise automatique après une coupure de Colab.

---

## 7. Chronologie du projet

| Période | Travail | Détails dans |
|---|---|---|
| 30/10 | Données (miroir Hugging Face, échantillon), exploration, référence classique, premiers CNN | `JOURNAL.md` semaine 1 |
| 31/10 | Jeu complet sur Colab (memmap, reprise), CNN complets, constat du changement de domaine | semaine 1 |
| 31/10 | Modèles de fondation figés + sonde linéaire (Phikon, Phikon-v2, Midnight, DINOv2) | semaine 2 |
| 01/11 | Stade pN : 57 patientes, lames lues à distance, règle de taille et forêt | semaine 2 |
| 01–02/11 | ABMIL hybride, ensemble de graines, combinaisons | semaine 2 |
| 05/11 | Point d'avancement pour l'équipe | `docs/point_avancement.pdf` |
| 12–13/11 | Notebook pédagogique, suivi avec l'enseignante | `notebooks/03_colab_cnn_explique.ipynb` |
| 25/11 | Augmentation HED, Grad-CAM, rapport final | semaine 4 |

---

## 8. Sources, outils et usage de l'IA

**Données et articles** : CAMELYON17 (Bándi et al., 2019) ; WILDS (Koh et al., 2021) ; Phikon (Filiot et al., 2023) ;
ABMIL (Ilse et al., 2018) ; augmentation HED (Tellez et al., 2019) ; Grad-CAM (Selvaraju et al., 2017).

**Outils** : PyTorch, torchvision, scikit-learn, scikit-image, Hugging Face (`huggingface_hub`, `transformers`),
Google Colab, LaTeX.

**Usage de l'IA** : une partie du code et de la rédaction a été produite avec l'aide d'un assistant d'IA (Claude) ;
l'équipe a relu et compris chaque partie, et cet usage est déclaré dans le rapport.
