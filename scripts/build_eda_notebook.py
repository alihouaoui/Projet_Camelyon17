"""Génère notebooks/01_exploration.ipynb (on garde le notebook versionné, ce script sert à le reconstruire)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# 01 — Exploration des patchs Camelyon17-WILDS

Objectifs : vérifier la correspondance images / étiquettes, mesurer la répartition des classes
et des patientes par hôpital, visualiser des patchs représentatifs et **quantifier les différences
de coloration entre hôpitaux** (changement de domaine).

Données : `data/processed/subset.parquet`, sous-échantillon représentatif tiré des 21 fichiers
(`scripts/fetch_sample.py`). Les conclusions sur les effectifs absolus sont donc à confirmer sur le jeu complet.""")

code("""import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from skimage.color import rgb2hed, rgb2hsv
from src.data import load_parquet, assign_splits, decode

FIG = Path('../reports/figures'); FIG.mkdir(parents=True, exist_ok=True)
CENTER_NAME = {0: 'C0 (train)', 1: 'C1 (val OOD)', 2: 'C2 (test OOD)', 3: 'C3 (train)', 4: 'C4 (train)'}
df = assign_splits(load_parquet(['../data/processed/subset.parquet']))
print(f'{len(df)} patchs, {df.patient_id.nunique()} patientes, {df.slide_id.nunique()} lames')""")

md("## 1. Correspondance images / étiquettes et format")
code("""shapes = pd.Series([decode(b).shape for b in df.img.sample(500, random_state=0)]).value_counts()
print('Formats observés :', shapes.to_dict())
print('Valeurs manquantes :', int(df.drop(columns='img').isna().sum().sum()))
print('Étiquettes :', df.label.value_counts().to_dict())""")

md("## 2. Répartition par hôpital, patiente et classe")
code("""summary = df.groupby('center').agg(patchs=('label', 'size'), part_tumeur=('label', 'mean'),
                                   patientes=('patient_id', 'nunique'), lames=('slide_id', 'nunique'))
summary.index = summary.index.map(CENTER_NAME)
summary.round(3)""")
code("""split_tab = pd.crosstab(df.split, df.label, margins=True).rename(columns={0: 'non tumoral', 1: 'tumoral'})
split_tab['patientes'] = df.groupby('split').patient_id.nunique()
split_tab""")
md("""**À retenir.** WILDS a été construit **équilibré** (environ autant de patchs tumoraux que non tumoraux).
Ce n'est pas la réalité clinique : sur une lame entière, le tissu tumoral représente souvent moins de 1 %
de la surface. Le déséquilibre se posera donc surtout au niveau des lames (semaine 3) et lors du choix du seuil.""")

code("""per_slide = df.groupby(['center', 'slide_id']).agg(n=('label', 'size'), tumeur=('label', 'mean')).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for c, g in per_slide.groupby('center'):
    ax[0].scatter(g.n, g.tumeur, label=CENTER_NAME[c], s=30)
ax[0].set(xlabel='patchs par lame', ylabel='part de patchs tumoraux', title='Hétérogénéité entre lames'); ax[0].legend(fontsize=8)
pd.crosstab(df.center.map(CENTER_NAME), df.label).plot.bar(ax=ax[1], rot=0)
ax[1].set(title='Patchs par centre et par classe', xlabel=''); ax[1].legend(['non tumoral', 'tumoral'])
plt.tight_layout(); plt.savefig(FIG / 'repartition.png', dpi=150)""")

md("## 3. Patchs représentatifs par classe et par hôpital")
code("""fig, axes = plt.subplots(10, 6, figsize=(9, 15))
for i, c in enumerate(range(5)):
    for j, lab in enumerate((0, 1)):
        g = df[(df.center == c) & (df.label == lab)]
        sample = g.sample(min(6, len(g)), random_state=1)
        for k in range(6):
            a = axes[2 * i + j, k]; a.axis('off')
            if k < len(sample):
                a.imshow(decode(sample.img.iloc[k]))
        axes[2 * i + j, 0].set_title(f"{CENTER_NAME[c]} — {'tumoral' if lab else 'non tumoral'}", loc='left', fontsize=8)
plt.tight_layout(); plt.savefig(FIG / 'galerie.png', dpi=150)""")
md("""**Observations.** Les patchs tumoraux montrent des cellules plus grandes, aux noyaux volumineux et clairs
avec nucléoles ; les patchs non tumoraux sont surtout des nappes de lymphocytes (petits noyaux ronds et sombres),
de la graisse (grands vides blancs) ou du stroma rose. **La teinte varie fortement d'un hôpital à l'autre** :
le centre 2 (test) est nettement plus violet et saturé, le centre 4 plus pâle, les centres 0, 1 et 3 plus roses.""")

md("## 4. Coloration : quantifier le changement de domaine")
code("""rng = np.random.default_rng(0)
samp = df.groupby(['center', 'label'], group_keys=False).apply(lambda g: g.sample(min(len(g), 400), random_state=0))
def color_stats(png):
    rgb = decode(png); hsv = rgb2hsv(rgb); hed = rgb2hed(rgb)
    return [*rgb.reshape(-1, 3).mean(0), hsv[..., 1].mean(), hed[..., 0].mean(), hed[..., 1].mean()]
cols = ['R', 'G', 'B', 'saturation', 'hematoxyline', 'eosine']
stats = pd.DataFrame([color_stats(b) for b in samp.img], columns=cols, index=samp.index)
stats[['center', 'label']] = samp[['center', 'label']]
stats.groupby(['center', 'label'])[cols].mean().round(3)""")
code("""fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
for a, col in zip(axes, ['R', 'B', 'hematoxyline', 'eosine']):
    data = [stats.loc[stats.center == c, col] for c in range(5)]
    a.boxplot(data, tick_labels=[CENTER_NAME[c].split()[0] for c in range(5)], showfliers=False); a.set_title(col)
plt.suptitle('Distribution des statistiques de couleur par hôpital (toutes classes)'); plt.tight_layout()
plt.savefig(FIG / 'couleur_par_centre.png', dpi=150)""")
code("""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = stats[cols].values
acc_center = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, stats.center, cv=5).mean()
acc_label = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, stats.label, cv=5).mean()
print(f"Avec 6 statistiques de couleur : centre prédit à {acc_center:.1%} (hasard 20 %), classe prédite à {acc_label:.1%} (hasard ~50 %)")""")
md("""**Résultat.** Avec 6 nombres de couleur, l'hôpital est reconnu dans environ 80 % des cas (hasard : 20 %),
et la projection t-SNE sépare nettement les centres 2 et 4 des centres 0, 1 et 3. Le centre 2, réservé au test,
a une coloration que l'on ne voit dans aucun centre d'entraînement sauf en partie le centre 4.

Si la couleur seule permet de reconnaître l'hôpital bien mieux que le hasard, un modèle peut apprendre
des raccourcis liés à la coloration plutôt qu'à la morphologie. C'est l'argument pour la normalisation ou
l'augmentation de couleur (HED), testées en semaine 2, et pour évaluer sur des hôpitaux jamais vus (C1, C2).""")

code("""from sklearn.manifold import TSNE
emb = TSNE(random_state=0, init='pca', perplexity=40).fit_transform((X - X.mean(0)) / X.std(0))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for c in range(5):
    m = stats.center.values == c; ax[0].scatter(*emb[m].T, s=4, label=CENTER_NAME[c])
for lab, name in ((0, 'non tumoral'), (1, 'tumoral')):
    m = stats.label.values == lab; ax[1].scatter(*emb[m].T, s=4, label=name)
ax[0].set_title('Couleur : coloré par hôpital'); ax[1].set_title('Couleur : coloré par classe')
for a in ax: a.legend(markerscale=4, fontsize=8); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(FIG / 'tsne_couleur.png', dpi=150)""")

md("## 5. Position des patchs dans une lame")
code("""slide = per_slide.sort_values('n').iloc[-1].slide_id
g = df[df.slide_id == slide]
plt.figure(figsize=(5, 5)); plt.scatter(g.x_coord, g.y_coord, c=g.label, cmap='coolwarm', s=6)
plt.gca().invert_yaxis(); plt.title(f'Lame {slide} : patchs échantillonnés (rouge = tumoral)')
plt.savefig(FIG / 'positions_lame.png', dpi=150)""")
md("""Les patchs tumoraux sont **spatialement regroupés** : deux patchs voisins d'une même lame se ressemblent
beaucoup. C'est pourquoi un découpage aléatoire par patch gonflerait artificiellement les scores ;
on découpe par patiente (`assign_splits`) et on vérifie l'absence de fuite.""")

md("""## 6. Synthèse pour la suite

1. Format homogène (96×96 RGB), étiquettes binaires, aucune valeur manquante.
2. Jeu de patchs équilibré par construction : le vrai déséquilibre apparaîtra au niveau lame.
3. Forte variabilité de couleur entre hôpitaux : à traiter (HED, normalisation) et à évaluer hors domaine.
4. Patchs spatialement corrélés : découpage par patiente obligatoire.""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_exploration.ipynb"
nbf.write(nb, out)
print(out)
