# Journal de bord

Une entrée par séance : ce qui a été fait, les décisions avec leur justification, les résultats, les questions ouvertes.

## Semaine 1 — 30/10/2026

### Fait
- Dépôt créé : `src/` (données, modèles, augmentations, métriques, caractéristiques), `scripts/`, `configs/`, `notebooks/`.
- Données : sous-échantillon représentatif de Camelyon17-WILDS tiré des 21 fichiers du miroir Hugging Face.
- Notebook d'exploration `01_exploration.ipynb`, figures dans `reports/figures/`.
- Référence classique (couleur + texture → régression logistique, forêt aléatoire) et CNN (ResNet18 pré-entraîné, petit CNN de zéro) : chaîne testée de bout en bout.
- Notebook `02_colab.ipynb` pour l'entraînement complet sur GPU (Colab).

### Décisions
| Décision | Justification |
|---|---|
| Miroir Hugging Face au lieu de CodaLab | Certificat HTTPS de CodaLab expiré ; effectifs du miroir identiques à WILDS (455 954 patchs). |
| Sous-échantillon local, entraînement complet sur Colab | 12 Go libres sur le Mac, CPU Intel : ~1 h 30 par epoch complète en local. |
| Découpage par patiente, centres 1 et 2 réservés | Patchs voisins très corrélés ; le split `id_val` de WILDS mélange des patchs des mêmes lames. Contrôle automatique `check_no_leakage`. |
| Seuil choisi sur `val_id` pour une sensibilité ≥ 95 % | Les faux négatifs (métastase ratée) coûtent plus cher que les faux positifs. Le seuil 0,5 est rapporté à côté. |
| Patchs gardés en 96×96 | Taille native WILDS ; agrandir à 224 multiplie le coût par ~5 pour un gain à mesurer (semaine 2). |

### Résultats (échantillon local de 40 254 patchs, à confirmer sur le jeu complet)

**Exploration** (`notebooks/01_exploration.ipynb`)
- 43 patientes, 50 lames, 5 centres ; patchs 96×96 RGB, aucune valeur manquante ; classes équilibrées par construction.
- La couleur seule (6 statistiques) reconnaît l'hôpital dans 80 % des cas (hasard 20 %). Le centre 2 (test) est nettement plus violet et saturé.
- Les patchs tumoraux sont regroupés dans l'espace de la lame : cela confirme qu'il faut découper par patiente.

**Premiers modèles** (12 000 patchs d'entraînement, 3 epochs, CPU/MPS ; seuil fixé sur `val_id` pour une sensibilité de 95 %)

| Modèle | AUC val_id | AUC val_ood (C1) | AUC test_ood (C2) | Sensibilité C2 au seuil fixé |
|---|---|---|---|---|
| Couleur + texture, rég. logistique | 0,971 | 0,889 | **0,623** | 0,71 |
| Couleur + texture, forêt aléatoire | 0,959 | 0,920 | 0,898 | 0,98 (spécificité 0,45) |
| Petit CNN (de zéro) | 0,967 | 0,926 | 0,934 | 0,99 (spécificité 0,54) |
| ResNet18 pré-entraîné | **0,995** | **0,977** | **0,968** | **0,72** (spécificité 0,99) |

**Constat principal : un bon AUC ne suffit pas.** Le ResNet18 classe bien les patchs du centre 2 (AUC 0,968), mais
le seuil réglé sur les hôpitaux d'entraînement ne se transpose pas : sa sensibilité chute de 95 % à 72 %, soit
1 109 patchs tumoraux manqués sur 3 927. Les probabilités sont décalées par la coloration du centre 2.
La régression logistique sur la couleur s'effondre (AUC 0,62), car elle s'appuie sur la teinte.
→ Priorités de la semaine 2 : augmentation HED ou normalisation de couleur, et calibration ou réglage du seuil par centre.

Limite : le centre 2 a été évalué à chaque epoch sur cet échantillon de mise au point. Sur le jeu complet, le choix
du modèle et du seuil se fera uniquement sur `val_id` et `val_ood`, et le test final sera lancé une seule fois.

### 31/10 — Passage au jeu complet sur Colab
- Plateforme : **Colab gratuit** (GPU T4, 2 cœurs CPU, 12,7 Go de RAM, 113 Go de disque), piloté depuis le Mac
  avec la CLI officielle `google-colab-cli` (`scripts/colab_launch.sh`, `scripts/colab_watch.sh`).
- Problème : les 455 954 images ne tiennent pas en RAM. Solution : conversion unique en images décodées sur disque
  (`images.npy`, 12,6 Go, 8 min sur Colab). Vérifié : pixels identiques au format d'origine.
- Problème : Google a coupé la première session après ~1 h de GPU ; modèle et évaluation du ResNet18 perdus
  (seuls les petits fichiers étaient rapatriés). Solution : `best.pt` et `last.pt` rapatriés sur le Mac toutes les
  4 min, renvoyés à Colab au relancement, reprise automatique.
- Problème : avec 2 cœurs CPU, la préparation des images bridait l'entraînement (~395 s/epoch). Solution :
  symétries et rotations faites par lot sur le GPU (vérifié identique à la version PIL) : ~250 s/epoch.
- Le run de 5 epochs coupé est gardé (`outputs_colab/resnet18_full_5ep_coupe/history.csv`) : l'AUC sur C1 baisse
  de 0,979 (epoch 1) à 0,961 (epoch 5). D'où 3 epochs pour le ResNet18 et 5 pour le petit CNN.
- Rigueur : le centre 2 n'est évalué qu'une fois, à la fin.

### 31/10 — Résultats sur le jeu complet (455 954 patchs)

Seuil fixé sur `val_id` pour une sensibilité de 95 %, puis appliqué tel quel. Référence classique entraînée sur
30 000 patchs et évaluée sur 20 000 patchs par split ; CNN évalués sur la totalité des splits.

| Modèle | AUC val_id | AUC C1 (val_ood) | AUC C2 (test) | Sensibilité C2 | Spécificité C2 | Tumeurs C2 manquées |
|---|---|---|---|---|---|---|
| Couleur + texture, rég. logistique | 0,973 | 0,838 | 0,660 | 0,63 | 0,62 | 37 % |
| Couleur + texture, forêt aléatoire | 0,965 | 0,901 | 0,896 | 0,99 | 0,36 | 1 % |
| Petit CNN (5 epochs) | 0,993 | 0,912 | 0,893 | 0,82 | 0,79 | 18 % |
| **ResNet18 pré-entraîné (3 epochs)** | **0,997** | **0,979** | **0,958** | 0,69 | **0,99** | **31 % (13 183 / 42 527)** |

**Constats**
1. **Le ResNet18 classe le mieux (AUC 0,958 sur C2), mais rate 31 % des tumeurs de C2 au seuil appris ailleurs.**
   Pour retrouver 95 % de sensibilité sur C2, il faudrait un seuil de 0,023 au lieu de 0,843 : ses probabilités
   sont fortement décalées vers « non tumoral » sur l'hôpital au rendu violet.
2. **Régler le seuil sur un autre hôpital inconnu (C1) aide peu** : 0,631 au lieu de 0,843, sensibilité C2 de 0,69 à 0,74.
   Le décalage est propre à chaque hôpital → piste : normaliser ou augmenter la couleur plutôt que régler le seuil.
3. **Plus on entraîne, plus on se spécialise** : AUC C1 de 0,983 (epoch 1) à 0,958 (epoch 3) pendant que val_id reste à 0,996.
   Le choix du modèle sur val_id (epoch 2) ne voit pas cette dégradation.
4. **Les modèles « couleur » ne transfèrent pas** (régression logistique : AUC 0,66 sur C2). La forêt aléatoire
   garde la sensibilité mais au prix de 64 % de faux positifs.

**Faiblesse de notre découpage, à corriger en semaine 2.** `val_id` ne contient que 5 patientes, et 14 208 de
ses 14 357 patchs tumoraux viennent de 2 patientes du centre 3 (C0 : 5 tumeurs, C4 : 144). Le seuil et le choix
du modèle reposent donc presque entièrement sur le centre 3. → Tirer les patientes de validation en stratifiant
sur la présence de tumeur, pour que chaque centre d'entraînement y contribue.

### Priorités semaine 2 (issues des résultats)
1. Validation stratifiée par patiente et présence de tumeur.
2. Augmentation HED et normalisation de couleur, comparées sur C1 puis une seule fois sur C2.
3. Choix de l'epoch sur val_ood (les checkpoints `epoch*.pt` sont gardés pour le faire a posteriori).
4. Grad-CAM sur les faux négatifs de C2.

### Questions pour l'enseignante
1. Une cohorte réduite de lames complètes (~10 patientes) suffit-elle pour l'étape stade pN ?
2. Un VAE (visualisation du changement de domaine) est-il bienvenu en complément ?
3. Le test final sur le centre 2 doit-il rester intouché jusqu'au rapport, comme prévu ?

### Prochaine étape (semaine 2)
Voir « Priorités semaine 2 » ci-dessus ; en parallèle, téléchargement et découpage des lames complètes.
