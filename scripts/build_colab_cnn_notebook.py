"""Génère notebooks/03_colab_cnn_explique.ipynb : notebook autonome à partager (données + CNN, tout expliqué).

Il ne dépend pas du dépôt : il télécharge lui-même un échantillon depuis Hugging Face, ce qui permet
de l'ouvrir directement dans Google Colab.
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

# ---------------------------------------------------------------------------------------------------------------
md("""# Projet CAMELYON17 — Détecter le cancer sur des images de ganglions avec un CNN

**Notebook d'équipe, à exécuter dans Google Colab** (menu *Exécution → Tout exécuter*, de préférence avec un GPU :
*Exécution → Modifier le type d'exécution → GPU T4*). Durée totale : environ 15 à 30 minutes (surtout le téléchargement).

Ce notebook reprend **le début du projet**, pas à pas :

| Partie | Contenu |
|---|---|
| 1 | Le problème médical et l'objectif |
| 2 | Les données : lesquelles, pourquoi celles-ci, comment on les récupère |
| 3 | Exploration des données (ce qu'on y voit, ce que ça implique) |
| 4 | Découpage entraînement / validation / test **sans fuite** |
| 5 | Le modèle : un CNN classique, couche par couche |
| 6 | La fonction de perte, l'optimiseur, l'entraînement |
| 7 | Les métriques : définitions et à quoi elles servent |
| 8 | Résultats et interprétation |
| 9 | Pour la suite : chacun son modèle |

> Tout est autonome : aucun fichier à fournir, les données sont téléchargées automatiquement.""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 1. Le problème médical et l'objectif

Dans le cancer du sein, on retire les **ganglions lymphatiques** de l'aisselle pour savoir si le cancer s'y est propagé
(on parle de **métastase**). Chaque ganglion est coupé en fines tranches, coloré, puis numérisé au microscope :
on obtient une **lame entière** (*whole slide image*), une image gigantesque (≈ 100 000 × 200 000 pixels, 1 à 3 Go).

La coloration utilisée est l'**H&E** (hématoxyline et éosine) :
- l'**hématoxyline** colore les **noyaux** des cellules en **violet / bleu** ;
- l'**éosine** colore le **cytoplasme** et les fibres en **rose**.

Un pathologiste parcourt la lame à la recherche de cellules tumorales. C'est long et fatigant, et les petites métastases
peuvent être ratées. **Objectif de cette première étape** : apprendre à un modèle à dire, pour chaque petit carré d'image
(un **patch** de 96 × 96 pixels), s'il contient du **tissu tumoral (1)** ou **non (0)**.

C'est une **classification binaire d'images**. Plus tard, on assemble les prédictions de tous les patchs d'une lame
pour obtenir une carte de chaleur, puis le stade de la patiente (étape 2 du projet, hors de ce notebook).

**Ce qui compte le plus** : une métastase ratée (faux négatif) est bien plus grave qu'une fausse alerte (faux positif),
car la patiente risque de ne pas recevoir le traitement dont elle a besoin. Cela va guider le choix des métriques.""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 2. Les données

### 2.1 Lesquelles ?
On utilise **Camelyon17-WILDS** (Koh et al., 2021), une version « prête à l'emploi » du challenge **CAMELYON17** (Bándi et al., 2019) :

- **455 954 patchs** couleur de **96 × 96 pixels** ;
- découpés dans **50 lames** de ganglions, provenant de **43 patientes** et de **5 hôpitaux** néerlandais (10 lames par hôpital) ;
- chaque patch a une **étiquette** : 1 si son centre tombe dans une zone tumorale **dessinée à la main par des pathologistes**, 0 sinon ;
- résolution : environ **1 µm par pixel**, un patch couvre donc ≈ 0,1 mm de tissu ;
- métadonnées : hôpital (`center`), patiente (`patient`), ganglion (`node`), position dans la lame (`x_coord`, `y_coord`).

### 2.2 Pourquoi ce jeu de données ?
1. **Problème clinique réel**, avec une vérité terrain fiable (contours tracés par des experts).
2. **Plusieurs hôpitaux** : chaque hôpital a son scanner et sa recette de coloration, donc des couleurs différentes.
   C'est l'occasion d'étudier le **changement de domaine** : un modèle qui marche dans ses hôpitaux d'entraînement
   marche-t-il encore dans un hôpital qu'il n'a jamais vu ? WILDS a été créé exactement pour ça.
3. **Déjà découpé en patchs** et équilibré : on peut entraîner un CNN directement, sans manipuler les lames de plusieurs Go.
4. **Références publiées** (challenge, benchmark WILDS) pour situer nos résultats.

### 2.3 Comment on les récupère ?
Le serveur officiel de WILDS (CodaLab) avait un certificat HTTPS expiré ; on utilise une **copie identique sur Hugging Face**
(`wltjr1007/Camelyon17-WILDS`, 21 fichiers *parquet*, 10,7 Go en tout, effectifs vérifiés identiques).

Pour que le notebook tourne vite, on ne télécharge **pas tout** : on lit à distance quelques **blocs de 100 patchs régulièrement
espacés dans chacun des 21 fichiers**. Comme les fichiers sont triés par hôpital, classe et lame, ce tirage couvre tous les
hôpitaux, toutes les lames et les deux classes. Avec `BLOCS_PAR_FICHIER = 10`, on obtient ≈ 16 500 patchs, de toutes les patientes (43) et toutes les lames (50).

> Sur le jeu complet (455 954 patchs, entraîné sur Colab), les résultats sont donnés en partie 8 pour comparaison.""")

code("""# Paramètres du notebook (modifiables)
BLOCS_PAR_FICHIER = 10   # 10 -> ~16 500 patchs (5 à 15 min de téléchargement). 20 -> ~33 000 patchs.
MODELE = 'petit_cnn'     # 'petit_cnn' (appris de zéro) ou 'resnet18' (pré-entraîné sur ImageNet)
EPOQUES = 8
TAILLE_LOT = 128         # nombre d'images vues avant chaque mise à jour des poids (batch)
TAUX_APPRENTISSAGE = 1e-3
SENSIBILITE_CIBLE = 0.95 # on règle le seuil pour détecter 95 % des tumeurs (voir partie 7)
GRAINE = 0""")

code("""import io, time, random
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import pyarrow as pa, pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from huggingface_hub import HfFileSystem

random.seed(GRAINE); np.random.seed(GRAINE); torch.manual_seed(GRAINE)
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print('Calcul sur :', device, '| PyTorch', torch.__version__)
if device == 'cpu':
    print('Pas de GPU : Exécution -> Modifier le type d\\'exécution -> GPU T4 (sinon tout marche, mais plus lentement).')""")

code("""%%time
REPO = 'datasets/wltjr1007/Camelyon17-WILDS/data'
COLS = ['patient', 'node', 'center', 'x_coord', 'y_coord', 'label', 'image']
fs = HfFileSystem()
fichiers = sorted(f for f in fs.ls(REPO, detail=False) if f.endswith('.parquet'))
tables = []
for f in fichiers:
    with fs.open(f, 'rb', block_size=4 * 2**20) as fh:
        pf = pq.ParquetFile(fh)
        n = pf.metadata.num_row_groups
        blocs = np.unique(np.linspace(0, n - 1, min(n, BLOCS_PAR_FICHIER)).round().astype(int)).tolist()
        tables.append(pf.read_row_groups(blocs, columns=COLS))
    print(f"{f.split('/')[-1]} : {tables[-1].num_rows} patchs")
t = pa.concat_tables(tables)
png = t.column('image').combine_chunks().field('bytes').to_pylist()
df = t.drop(['image']).to_pandas()
df['patient_id'] = df.patient.astype(int)
df['lame'] = df.patient.astype(str) + '_' + df.node.astype(str)

# décodage des PNG en un seul tableau (N, 96, 96, 3) d'entiers 0-255
images = np.stack([np.asarray(Image.open(io.BytesIO(b)).convert('RGB')) for b in png])
del png, t, tables
print(f"\\n{len(df)} patchs, {df.patient_id.nunique()} patientes, {df.lame.nunique()} lames ; images : {images.shape}, "
      f"{images.nbytes / 1e9:.2f} Go en mémoire")""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 3. Exploration des données

Avant tout modèle, on vérifie la qualité des données et on cherche ce qui pourrait piéger un modèle.

### 3.1 Format, valeurs manquantes, classes""")

code("""print('Taille des images :', images.shape[1:], '| type :', images.dtype)
print('Valeurs manquantes dans les métadonnées :', int(df.isna().sum().sum()))
print('Classes :', df.label.value_counts().rename({0: 'non tumoral (0)', 1: 'tumoral (1)'}).to_dict())
resume = df.groupby('center').agg(patchs=('label', 'size'), part_tumorale=('label', 'mean'),
                                  patientes=('patient_id', 'nunique'), lames=('lame', 'nunique'))
resume.index = [f'hôpital {c}' for c in resume.index]
resume.round(2)""")

md("""**À retenir.**
- Toutes les images ont le même format (96 × 96 × 3), aucune valeur manquante.
- Les deux classes sont **à peu près équilibrées** : c'est un choix des créateurs de WILDS. Dans la réalité, sur une lame
  entière, le tissu tumoral représente souvent **moins de 1 %** de la surface. Le déséquilibre reviendra quand on passera aux lames.
- La part de patchs tumoraux **varie selon l'hôpital** et selon la lame (certaines lames ont beaucoup de tumeur, d'autres très peu).""")

md("### 3.2 À quoi ressemblent les patchs ?")
code("""fig, axes = plt.subplots(5, 8, figsize=(13, 8.5))
for c in range(5):
    for lab in (0, 1):
        g = df[(df.center == c) & (df.label == lab)]
        idx = g.sample(min(4, len(g)), random_state=1).index
        for k, i in enumerate(idx):
            a = axes[c, lab * 4 + k]; a.imshow(images[i]); a.set_xticks([]); a.set_yticks([])
    axes[c, 0].set_ylabel(f'hôpital {c}', fontsize=11)
axes[0, 1].set_title('NON tumoral (0)', fontsize=12, loc='left'); axes[0, 5].set_title('TUMORAL (1)', fontsize=12, loc='left')
plt.tight_layout(); plt.show()""")
md("""**Ce qu'on observe.**
- **Non tumoral** : surtout des nappes de **lymphocytes** (petits noyaux ronds, sombres et serrés), de la **graisse** (grands vides blancs)
  ou du tissu de soutien rose.
- **Tumoral** : des cellules **plus grandes**, aux **noyaux volumineux, plus clairs et irréguliers**, disposées de façon désorganisée.
- **La couleur change beaucoup d'un hôpital à l'autre**, à classe égale : l'hôpital 2 est nettement plus **violet et saturé**,
  l'hôpital 4 plus pâle. Un humain s'adapte ; un modèle, pas forcément.""")

md("""### 3.3 Mesurer le changement de domaine : la couleur suffit-elle à reconnaître l'hôpital ?
On résume chaque patch par 6 nombres de couleur (moyennes rouge / vert / bleu, saturation, quantité d'hématoxyline et d'éosine),
puis on demande à une simple régression logistique de deviner **l'hôpital**. Si elle y arrive bien mieux que le hasard (20 %),
c'est que les hôpitaux ont des « signatures » de couleur qu'un CNN pourrait apprendre **à la place** de la forme des cellules.""")
code("""from skimage.color import rgb2hed, rgb2hsv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ech = df.groupby(['center', 'label'], group_keys=False).apply(lambda g: g.sample(min(len(g), 300), random_state=0)).index
def couleur(img):
    x = img / 255.0; hed = rgb2hed(x)
    return [*x.reshape(-1, 3).mean(0), rgb2hsv(x)[..., 1].mean(), hed[..., 0].mean(), hed[..., 1].mean()]
noms = ['rouge', 'vert', 'bleu', 'saturation', 'hématoxyline', 'éosine']
C = pd.DataFrame([couleur(images[i]) for i in ech], columns=noms, index=ech)
C['hôpital'] = df.loc[ech, 'center'].values

fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for a, col in zip(axes, ['rouge', 'bleu', 'saturation', 'hématoxyline']):
    a.boxplot([C.loc[C['hôpital'] == c, col] for c in range(5)], showfliers=False)
    a.set_xticklabels([f'H{c}' for c in range(5)]); a.set_title(col)
plt.suptitle('Couleur des patchs selon l\\'hôpital'); plt.tight_layout(); plt.show()

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
acc = cross_val_score(clf, C[noms], C['hôpital'], cv=5).mean()
print(f"Avec 6 nombres de couleur, l'hôpital est reconnu dans {acc:.0%} des cas (hasard : 20 %).")""")
md("""**Conséquence.** L'hôpital se reconnaît très bien à la couleur seule (≈ 80 % sur nos données). Un modèle entraîné sur certains
hôpitaux risque donc de mal se comporter sur un hôpital à la coloration différente. C'est pourquoi on va **évaluer sur des hôpitaux
jamais vus** pendant l'entraînement.""")

md("""### 3.4 Les patchs d'une même lame sont-ils indépendants ?""")
code("""lame = df.lame.value_counts().index[0]
g = df[df.lame == lame]
plt.figure(figsize=(5, 5)); plt.scatter(g.x_coord, g.y_coord, c=g.label, cmap='coolwarm', s=8)
plt.gca().invert_yaxis(); plt.title(f'Lame {lame} : position des patchs (rouge = tumoral)'); plt.show()""")
md("""Les patchs tumoraux sont **regroupés** au même endroit : deux patchs voisins se ressemblent énormément.
Si on mélangeait les patchs au hasard entre entraînement et test, le modèle verrait au test des quasi-copies d'images
apprises : le score serait **artificiellement gonflé**. C'est une **fuite de données**. D'où la règle de la partie suivante.""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 4. Découpage entraînement / validation / test, sans fuite

**Règle d'or : on découpe par patiente, jamais par patch.** Une patiente est entièrement d'un seul côté.

| Ensemble | Contenu | À quoi il sert |
|---|---|---|
| `train` | hôpitaux 0, 3, 4 — 80 % des patientes | apprendre les poids du modèle |
| `val_id` | hôpitaux 0, 3, 4 — les 20 % autres patientes | choisir la meilleure époque et le **seuil de décision** |
| `val_ood` | hôpital 1 (jamais vu) | vérifier la généralisation à un nouvel hôpital |
| `test_ood` | hôpital 2 (jamais vu) | **test final**, regardé une seule fois à la fin |

*id* = *in distribution* (mêmes hôpitaux que l'entraînement) ; *ood* = *out of distribution* (hôpital inconnu).

On n'utilise pas le découpage `id_val` officiel de WILDS : il reprend des patchs **des mêmes lames** que l'entraînement (fuite).""")
code("""rng = np.random.default_rng(GRAINE)
df['split'] = 'train'
for c in (0, 3, 4):
    pts = np.sort(df.loc[df.center == c, 'patient_id'].unique())
    val = rng.choice(pts, size=max(1, round(0.2 * len(pts))), replace=False)
    df.loc[(df.center == c) & df.patient_id.isin(val), 'split'] = 'val_id'
df.loc[df.center == 1, 'split'] = 'val_ood'
df.loc[df.center == 2, 'split'] = 'test_ood'

# contrôle automatique : aucune patiente dans deux ensembles
assert (df.groupby('patient_id').split.nunique() == 1).all(), 'Fuite : une patiente est dans deux ensembles !'
tab = pd.crosstab(df.split, df.label).rename(columns={0: 'non tumoral', 1: 'tumoral'})
tab['patientes'] = df.groupby('split').patient_id.nunique()
tab.loc[['train', 'val_id', 'val_ood', 'test_ood']]""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 5. Le modèle : un CNN classique

### 5.1 Pourquoi un CNN ?
Un **réseau de neurones convolutif** (CNN) est conçu pour les images. Au lieu de regarder chaque pixel séparément, il fait glisser
de petits **filtres** (3 × 3 pixels) sur l'image. Chaque filtre apprend à détecter un motif local : un bord, une texture,
puis, en empilant les couches, des formes de plus en plus complexes (un noyau, un amas de cellules).
Les mêmes filtres sont appliqués partout dans l'image : un noyau tumoral est reconnu où qu'il soit.

### 5.2 Les briques
| Brique | Rôle |
|---|---|
| **Convolution 3×3** | fait glisser des filtres appris ; produit des « cartes de caractéristiques » |
| **Normalisation de lot** (BatchNorm) | recentre et réduit les valeurs à chaque couche : entraînement plus stable et plus rapide |
| **ReLU** : $\\max(0, x)$ | non-linéarité : sans elle, empiler des couches reviendrait à une seule opération linéaire |
| **Max-pooling 2×2** | garde le maximum de chaque carré 2×2 : divise la taille par 2, rend le modèle tolérant aux petits décalages |
| **Pooling global** | moyenne de chaque carte sur toute l'image : un nombre par filtre |
| **Dropout** (30 %) | éteint au hasard 30 % des neurones pendant l'entraînement : limite le sur-apprentissage |
| **Couche linéaire** | combine les caractéristiques en 2 scores (« logits ») : non tumoral / tumoral |

### 5.3 Architecture du petit CNN (appris de zéro)
```
Image 3 × 96 × 96
 → [Conv 3×3, 32 filtres → BatchNorm → ReLU → MaxPool]   → 32 × 48 × 48
 → [Conv 3×3, 64 filtres → BatchNorm → ReLU → MaxPool]   → 64 × 24 × 24
 → [Conv 3×3, 128 filtres → BatchNorm → ReLU → MaxPool]  → 128 × 12 × 12
 → [Conv 3×3, 256 filtres → BatchNorm → ReLU → MaxPool]  → 256 × 6 × 6
 → Pooling global → 256 → Dropout 0,3 → Linéaire → 2 scores
```
≈ 390 000 paramètres. Option `MODELE = 'resnet18'` : un réseau résiduel de 18 couches (11 millions de paramètres)
**pré-entraîné sur ImageNet** (photos ordinaires), dont on remplace la dernière couche et qu'on ré-entraîne entièrement (*fine-tuning*).""")
code("""class PetitCNN(nn.Module):
    def __init__(self, largeur=32):
        super().__init__()
        couches, c_in = [], 3
        for c_out in (largeur, largeur * 2, largeur * 4, largeur * 8):
            couches += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                        nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.caracteristiques = nn.Sequential(*couches)
        self.tete = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c_in, 2))

    def forward(self, x):
        return self.tete(self.caracteristiques(x))

def construire_modele(nom):
    if nom == 'petit_cnn':
        return PetitCNN()
    import torchvision
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

modele = construire_modele(MODELE).to(device)
print(modele)
print(f"\\nNombre de paramètres appris : {sum(p.numel() for p in modele.parameters()):,}".replace(',', ' '))""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 6. Fonction de perte, optimiseur et entraînement

### 6.1 Des scores aux probabilités : le *softmax*
Le modèle sort 2 scores $(z_0, z_1)$. Le **softmax** les transforme en probabilités qui somment à 1 :
$$p_k = \\frac{e^{z_k}}{e^{z_0} + e^{z_1}}, \\qquad p_1 = P(\\text{tumoral} \\mid \\text{image}).$$

### 6.2 La fonction de perte : l'entropie croisée
La **fonction de perte** mesure à quel point le modèle se trompe ; l'entraînement cherche à la **minimiser**.
Pour la classification, on utilise l'**entropie croisée** (*cross-entropy*). Pour un patch de vraie classe $y \\in \\{0, 1\\}$ :
$$\\ell = -\\log p_y \\qquad\\text{soit}\\qquad \\ell = -\\big[\\,y \\log p_1 + (1 - y) \\log (1 - p_1)\\,\\big].$$
Sur un lot de $N$ images, on prend la moyenne $\\mathcal{L} = \\frac{1}{N}\\sum_i \\ell_i$.

**Intuition.** Si le patch est tumoral et que le modèle donne $p_1 = 0{,}9$ : perte $= -\\log 0{,}9 \\approx 0{,}11$ (faible).
S'il donne $p_1 = 0{,}1$ : perte $= -\\log 0{,}1 \\approx 2{,}3$ (forte). Une erreur **sûre d'elle** coûte très cher : le modèle est
poussé à être confiant quand il a raison et prudent quand il hésite.

**Pourquoi pas de pondération des classes ?** Les deux classes sont à peu près équilibrées ici. Si elles ne l'étaient pas, on pourrait
pondérer : $\\ell = -w_y \\log p_y$ avec un poids plus fort pour la classe rare (`nn.CrossEntropyLoss(weight=...)`).

### 6.3 L'optimiseur : comment on ajuste les poids
1. On passe un **lot** (*batch*) de 128 images dans le modèle (*forward*) ;
2. on calcule la perte $\\mathcal{L}$ ;
3. la **rétropropagation** (*backward*) calcule le **gradient** : dans quelle direction modifier chaque poids pour faire baisser la perte ;
4. l'optimiseur **AdamW** fait un petit pas dans cette direction. Adam adapte la taille du pas à chaque poids ; le « W »
   (*weight decay*) pousse les poids vers 0, ce qui limite le sur-apprentissage.
5. Le **taux d'apprentissage** (taille du pas) suit un calendrier *OneCycle* : il monte puis redescend, ce qui stabilise la fin.

Une **époque** = un passage sur toutes les images d'entraînement.

### 6.4 Augmentation de données
Un tissu au microscope n'a pas de « haut » ni de « bas » : on peut **retourner** et **tourner** les patchs de 90° sans changer
leur étiquette. À chaque lot, on applique ces transformations au hasard : le modèle voit plus de variété et apprend la forme
des cellules plutôt que leur orientation.

### 6.5 Normalisation des entrées
Les pixels (0–255) sont ramenés à 0–1, puis centrés-réduits canal par canal avec la moyenne et l'écart-type calculés
**sur l'entraînement uniquement** (les calculer sur le test serait une fuite).""")
code("""idx = {s: np.where(df.split.values == s)[0] for s in ['train', 'val_id', 'val_ood', 'test_ood']}
y_all = df.label.values
moy = torch.tensor(images[idx['train'][::10]].reshape(-1, 3).mean(0) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)
ect = torch.tensor(images[idx['train'][::10]].reshape(-1, 3).std(0) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)

def preparer(lot_idx, augmenter=False):
    x = torch.from_numpy(images[lot_idx]).to(device).permute(0, 3, 1, 2).float() / 255   # (N, 3, 96, 96)
    if augmenter:
        if random.random() < 0.5: x = x.flip(3)                      # miroir horizontal
        if random.random() < 0.5: x = x.flip(2)                      # miroir vertical
        x = torch.rot90(x, random.randint(0, 3), dims=(2, 3))         # rotation de 0, 90, 180 ou 270°
    return (x - moy) / ect

@torch.no_grad()
def predire(modele, ens, taille=512):
    modele.eval()
    out = [F.softmax(modele(preparer(ens[i:i + taille])), 1)[:, 1].float().cpu().numpy() for i in range(0, len(ens), taille)]
    return np.concatenate(out)""")
code("""from sklearn.metrics import roc_auc_score

perte_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-4)
n_lots = int(np.ceil(len(idx['train']) / TAILLE_LOT))
calendrier = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=TAUX_APPRENTISSAGE, total_steps=EPOQUES * n_lots)

historique, meilleur_auc, meilleurs_poids = [], -1, None
for ep in range(1, EPOQUES + 1):
    t0 = time.time(); modele.train(); pertes = []
    ordre = np.random.permutation(idx['train'])
    for b in range(n_lots):
        lot = np.sort(ordre[b * TAILLE_LOT:(b + 1) * TAILLE_LOT])
        x, y = preparer(lot, augmenter=True), torch.from_numpy(y_all[lot]).to(device)
        perte = perte_fn(modele(x), y)          # 1-2. forward + perte
        opt.zero_grad(); perte.backward()        # 3. gradients
        opt.step(); calendrier.step()            # 4-5. mise à jour des poids
        pertes.append(perte.item())
    # suivi : on ne regarde PAS l'hôpital 2 (test) pendant l'entraînement
    p_val = predire(modele, idx['val_id']); p_ood = predire(modele, idx['val_ood'])
    auc_val, auc_ood = roc_auc_score(y_all[idx['val_id']], p_val), roc_auc_score(y_all[idx['val_ood']], p_ood)
    historique.append({'époque': ep, 'perte_train': np.mean(pertes), 'AUC val_id': auc_val, 'AUC val_ood (H1)': auc_ood})
    if auc_val > meilleur_auc:   # le choix de l'époque se fait sur val_id uniquement
        meilleur_auc = auc_val; meilleurs_poids = {k: v.detach().clone() for k, v in modele.state_dict().items()}
    print(f"époque {ep}/{EPOQUES} — perte {np.mean(pertes):.3f} — AUC val_id {auc_val:.3f} — AUC hôpital 1 {auc_ood:.3f} "
          f"— {time.time() - t0:.0f} s")
modele.load_state_dict(meilleurs_poids)
H = pd.DataFrame(historique).set_index('époque')""")
code("""fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
H.perte_train.plot(ax=ax[0], marker='o', title='Perte d\\'entraînement (entropie croisée)')
H[['AUC val_id', 'AUC val_ood (H1)']].plot(ax=ax[1], marker='o', title='AUC sur la validation')
ax[0].set_ylabel('perte'); ax[1].set_ylabel('AUC'); plt.tight_layout(); plt.show()
print(f"Époque retenue (meilleure AUC sur val_id) : {H['AUC val_id'].idxmax()}")""")
md("""**Lecture.** La perte doit baisser au fil des époques. Si l'AUC sur `val_id` stagne pendant que celle de l'hôpital 1 baisse,
le modèle **se spécialise** sur ses hôpitaux d'entraînement : c'est le sur-apprentissage *au domaine*.""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 7. Les métriques : définitions et à quoi elles servent

Le modèle donne une **probabilité** $p_1$. Pour décider « tumoral / non tumoral », il faut un **seuil** $s$ : tumoral si $p_1 \\ge s$.

### 7.1 La matrice de confusion
|  | prédit non tumoral | prédit tumoral |
|---|---|---|
| **vraiment non tumoral** | VN (vrai négatif) | FP (faux positif = fausse alerte) |
| **vraiment tumoral** | FN (faux négatif = **tumeur ratée**) | VP (vrai positif) |

### 7.2 Les métriques
| Métrique | Formule | Question à laquelle elle répond | Pourquoi on la regarde |
|---|---|---|---|
| **Sensibilité** (rappel) | $\\frac{VP}{VP + FN}$ | Parmi les vraies tumeurs, combien sont trouvées ? | **Priorité n° 1** : une tumeur ratée = une patiente sous-traitée |
| **Spécificité** | $\\frac{VN}{VN + FP}$ | Parmi les tissus sains, combien sont bien écartés ? | Trop de fausses alertes = trop de relecture pour le pathologiste |
| **Précision** (valeur prédictive positive) | $\\frac{VP}{VP + FP}$ | Quand le modèle dit « tumoral », a-t-il raison ? | Confiance dans une alerte |
| **Exactitude** (*accuracy*) | $\\frac{VP + VN}{\\text{total}}$ | Part de bonnes réponses | Trompeuse si les classes sont déséquilibrées |
| **F1** | $\\frac{2 \\cdot \\text{précision} \\cdot \\text{sensibilité}}{\\text{précision} + \\text{sensibilité}}$ | Compromis précision / sensibilité | Un seul nombre, utile pour comparer |
| **AUC ROC** | aire sous la courbe ROC | Probabilité qu'un patch tumoral pris au hasard ait un score plus élevé qu'un patch sain pris au hasard | Mesure la capacité à **classer**, **sans choisir de seuil**. 0,5 = hasard, 1 = parfait |
| **AUC PR** | aire sous la courbe précision–rappel | Idem, centrée sur la classe positive | Plus informative quand les tumeurs sont rares |

La **courbe ROC** trace la sensibilité en fonction de $(1 - \\text{spécificité})$ quand on fait varier le seuil de 1 à 0.

### 7.3 Le choix du seuil
Le seuil 0,5 n'a rien de spécial. Comme rater une tumeur est grave, on **fixe le seuil sur `val_id` pour détecter 95 % des tumeurs**,
puis on l'applique **tel quel** aux hôpitaux 1 et 2. C'est le test le plus réaliste : à l'hôpital, on ne peut pas régler le seuil sur
les patientes qu'on est en train de diagnostiquer. Le seuil 0,5 est donné à côté pour comparaison.""")
code("""from sklearn.metrics import confusion_matrix, average_precision_score, roc_curve

def seuil_pour_sensibilite(y, p, cible=SENSIBILITE_CIBLE):
    pos = np.sort(p[y == 1])                      # scores des vrais patchs tumoraux, du plus petit au plus grand
    return float(pos[int(np.floor((1 - cible) * len(pos)))])

def metriques(y, p, s):
    vn, fp, fn, vp = confusion_matrix(y, (p >= s).astype(int), labels=[0, 1]).ravel()
    sens, spec = vp / (vp + fn), vn / (vn + fp)
    prec = vp / (vp + fp) if vp + fp else np.nan
    return {'AUC ROC': roc_auc_score(y, p), 'AUC PR': average_precision_score(y, p), 'seuil': s,
            'sensibilité': sens, 'spécificité': spec, 'précision': prec, 'exactitude': (vp + vn) / len(y),
            'F1': 2 * prec * sens / (prec + sens), 'tumeurs ratées (FN)': int(fn), 'fausses alertes (FP)': int(fp)}

P = {s: predire(modele, idx[s]) for s in idx if s != 'train'}   # l'hôpital 2 (test) est évalué ici, une seule fois
Y = {s: y_all[idx[s]] for s in P}
s95 = seuil_pour_sensibilite(Y['val_id'], P['val_id'])
print(f"Seuil choisi sur val_id pour {SENSIBILITE_CIBLE:.0%} de sensibilité : {s95:.3f}")
lignes = []
for nom_seuil, s in [('sensibilité 95 % (val_id)', s95), ('0,5', 0.5)]:
    for ens in ['val_id', 'val_ood', 'test_ood']:
        lignes.append({'seuil utilisé': nom_seuil, 'ensemble': ens, **metriques(Y[ens], P[ens], s)})
R = pd.DataFrame(lignes).set_index(['seuil utilisé', 'ensemble'])
R.style.format(precision=3)""")

# ---------------------------------------------------------------------------------------------------------------
md("## 8. Résultats et interprétation")
code("""fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
noms_ens = {'val_id': 'val_id (H0, H3, H4)', 'val_ood': 'hôpital 1', 'test_ood': 'hôpital 2 (test)'}
for ens in P:
    fpr, tpr, _ = roc_curve(Y[ens], P[ens]); axes[0].plot(fpr, tpr, label=f"{noms_ens[ens]} — AUC {roc_auc_score(Y[ens], P[ens]):.3f}")
axes[0].plot([0, 1], [0, 1], 'k:'); axes[0].set(xlabel='1 - spécificité', ylabel='sensibilité', title='Courbes ROC'); axes[0].legend(fontsize=8)
for a, ens in zip(axes[1:], P):
    m = confusion_matrix(Y[ens], (P[ens] >= s95).astype(int), labels=[0, 1])
    a.imshow(m, cmap='Blues')
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{m[i, j]}\\n{['VN', 'FP'][j] if i == 0 else ['FN', 'VP'][j]}", ha='center', va='center',
                   color='white' if m[i, j] > m.max() / 2 else 'black')
    a.set_xticks([0, 1], ['prédit 0', 'prédit 1']); a.set_yticks([0, 1], ['vrai 0', 'vrai 1'])
    a.set_title(f'{noms_ens[ens]}\\nseuil {s95:.2f}')
plt.tight_layout(); plt.show()""")
code("""# Où le modèle se trompe-t-il sur l'hôpital 2 ? (tumeurs ratées et fausses alertes les plus « sûres d'elles »)
ens = 'test_ood'; p, y, ii = P[ens], Y[ens], idx[ens]
fn = ii[(y == 1)][np.argsort(p[y == 1])[:6]]; fp = ii[(y == 0)][np.argsort(-p[y == 0])[:6]]
fig, axes = plt.subplots(2, 6, figsize=(13, 4.8))
for r, (sel, titre) in enumerate([(fn, 'tumeur ratée'), (fp, 'fausse alerte')]):
    for k, i in enumerate(sel):
        a = axes[r, k]; a.imshow(images[i]); a.axis('off')
        a.set_title(f"{titre}\\np = {p[np.where(ii == i)[0][0]]:.2f}", fontsize=8)
plt.suptitle('Hôpital 2 : erreurs les plus franches'); plt.tight_layout(); plt.show()""")
md("""### Comment lire ces résultats
1. **Comparer `val_id` aux hôpitaux 1 et 2** : l'AUC baisse dès qu'on change d'hôpital. C'est le **changement de domaine** mesuré
   en partie 3 : le modèle a appris en partie des couleurs propres à ses hôpitaux d'entraînement.
2. **Regarder la sensibilité au seuil fixé sur `val_id`** : sur les hôpitaux inconnus, elle tombe en dessous des 95 % visés.
   Les probabilités sont **décalées** par la coloration, si bien qu'un seuil réglé ailleurs rate davantage de tumeurs.
   Un bon AUC ne garantit donc pas un modèle utilisable tel quel dans un nouvel hôpital.
3. **Le seuil 0,5 serait pire** pour notre priorité : il rate encore plus de tumeurs (comparer les FN dans le tableau).
4. Les chiffres viennent d'un **échantillon** (≈ 16 500 patchs, quelques patientes par hôpital) et d'un petit modèle entraîné
   quelques minutes : ils varient d'une exécution à l'autre. Les tendances comptent plus que les décimales.

### Pour comparaison : nos résultats sur le jeu complet (455 954 patchs, Colab)
Seuil fixé sur `val_id` pour 95 % de sensibilité, puis appliqué tel quel :

| Modèle | AUC val_id | AUC hôpital 1 | AUC hôpital 2 | Sensibilité hôpital 2 | Spécificité hôpital 2 |
|---|---|---|---|---|---|
| Couleur + texture, régression logistique | 0,973 | 0,838 | 0,660 | 0,63 | 0,62 |
| Couleur + texture, forêt aléatoire | 0,965 | 0,901 | 0,896 | 0,99 | 0,36 |
| Petit CNN (5 époques) | 0,993 | 0,912 | 0,893 | 0,82 | 0,79 |
| ResNet18 pré-entraîné (3 époques) | 0,997 | 0,979 | 0,958 | **0,69** | 0,99 |

**Le constat principal** : le ResNet18 classe le mieux (AUC 0,958 sur l'hôpital 2), mais au seuil appris ailleurs il **rate 31 % des
tumeurs** de cet hôpital (13 183 patchs sur 42 527). Ses probabilités sont décalées par la coloration : pour retrouver 95 % de
sensibilité, il faudrait un seuil de 0,023 au lieu de 0,843. Le problème n'est pas la capacité à reconnaître les tumeurs, mais le
**changement de domaine**.""")

# ---------------------------------------------------------------------------------------------------------------
md("""## 9. Pour la suite : chacun son modèle

Ce notebook sert de **base commune** : mêmes données, même découpage par patiente, mêmes métriques, même règle de seuil.
Chacun peut tester une idée en modifiant **seulement** la partie 5 (le modèle) ou la partie 6 (l'entraînement), puis comparer
avec le tableau de la partie 7.

Pistes (à se répartir) :
- **Augmentation de couleur** (perturbation HED, *color jitter*) ou **normalisation de coloration** (Macenko, Reinhard) :
  attaque directement le changement de domaine.
- **Autres CNN pré-entraînés** : ResNet50, EfficientNet, DenseNet (`torchvision.models`).
- **Modèles de fondation en pathologie** (Phikon, UNI…) : un réseau figé pré-entraîné sur des millions d'images de tissus,
  plus une régression logistique.
- **Images agrandies** (96 → 224 pixels) pour les réseaux pré-entraînés sur ImageNet.
- **Interprétabilité** : Grad-CAM pour voir quelles zones du patch font décider le modèle.

**Règles communes pour que les comparaisons soient honnêtes**
1. Ne jamais choisir un modèle, une époque ou un seuil en regardant l'hôpital 2.
2. Toujours rapporter la sensibilité au seuil fixé sur `val_id`, pas seulement l'AUC.
3. Garder la même graine (`GRAINE = 0`) et le même découpage.""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_colab_cnn_explique.ipynb"
nbf.write(nb, out)
print(out)
