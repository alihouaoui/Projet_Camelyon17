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

## Semaine 2 — 31/10/2026 : modèles de fondation en pathologie (figés + sonde linéaire)

**Question posée** : un modèle pré-entraîné sur de la pathologie rend-il le seuil transférable à un nouvel hôpital ?
Plan critiqué au préalable par 5 relecteurs (pertinence, statistique, résolution, calcul, contamination).

**Méthode** (`src/foundation.py`, `scripts/extract_features.py`, `scripts/linear_probe.py`, `scripts/compare.py`)
- Modèles téléchargeables sans jeton, aucun pré-entraîné sur CAMELYON d'après leurs model cards : Phikon (ViT-B, TCGA),
  Phikon-v2 (ViT-L, TCGA + CPTAC + GTEx), Midnight-12k (ViT-g, TCGA) ; contrôle : DINOv2-base (images naturelles).
- Patchs 96×96 (~1 µm/pixel) agrandis à 224×224 ; caractéristiques figées (CLS ; CLS + moyenne pour Midnight).
- Régression logistique. **Correction du défaut de val_id** : régularisation choisie par validation croisée groupée
  par patiente (5 plis) sur les centres 0, 3, 4 ; seuil de sensibilité 95 % fixé sur les prédictions hors-pli.
- Comparaison avec **IC 95 % par bootstrap des patientes** (mêmes tirages pour tous les modèles).
- Calcul : Colab T4, ~2 h 30 (Phikon ~20 min, DINOv2 ~25 min, Phikon-v2 ~70 min, Midnight 25 img/s → 100 000 patchs).

**Résultats sur l'hôpital de test C2** (9 patientes ; seuil appris sans C1 ni C2)

| Modèle | AUC [IC 95 %] | Sensibilité [IC 95 %] | Spécificité | Tumeurs ratées |
|---|---|---|---|---|
| Petit CNN | 0,893 [0,74–0,94] | 0,82 [0,48–0,92] | 0,79 | 7 808 |
| ResNet18 fine-tuné | 0,958 [0,89–0,98] | 0,69 [0,28–0,80] | 0,99 | 13 183 |
| DINOv2 générique, figé | 0,977 [0,95–0,98] | 0,96 [0,90–0,97] | 0,86 | 1 792 |
| Phikon, figé | 0,990 [0,97–0,99] | 0,98 [0,95–0,99] | 0,83 | 705 |
| Phikon-v2, figé | 0,989 [0,96–1,00] | 0,97 [0,90–0,99] | 0,95 | 1 322 |
| Midnight-12k, figé (100 k patchs) | **0,996** [0,99–1,00] | 0,96 [0,90–0,98] | **1,00** | 391 sur 18 655 patchs |

Figure : `reports/figures/sensibilite_c2.png` ; tableau complet (C1 et C2) : `reports/comparaison_modeles.csv`.

**Constats**
1. **Oui, le seuil se transfère** avec les modèles de pathologie : réglé pour 95 % sur les hôpitaux d'entraînement,
   il donne 95–98 % sur C1 et C2. Le ResNet18 tombe à 69 %, avec un IC de 28 à 80 % : très instable selon les patientes.
2. Une simple régression logistique sur des caractéristiques figées bat le ResNet18 entraîné de bout en bout.
3. Le pré-entraînement en pathologie compte : DINOv2 générique transfère aussi son seuil, mais avec une spécificité de 0,86.
4. Entre Phikon, Phikon-v2 et Midnight, les IC se chevauchent : pas de vainqueur net sur la sensibilité.
   Midnight a le meilleur compromis (AUC 0,996, quasi aucune fausse alerte).

**Limites**
- Le seuil du ResNet18 vient de val_id (5 patientes), celui des sondes de la validation croisée : la différence de règle
  explique une partie de l'écart, pas tout (l'AUC des sondes est aussi meilleure).
- C2 a été regardé pour 6 modèles : résultats descriptifs. Le modèle final doit être choisi sur la validation croisée et C1,
  pas sur C2. Sur ces critères : Midnight (AUC hors-pli 0,996), puis Phikon / Phikon-v2 (0,991–0,993).
- Midnight : sous-échantillon de 100 000 patchs, et régularisation choisie au bord de la grille (0,1) → grille à élargir.
- Seulement 8–9 patientes par hôpital externe : IC larges, à rappeler à chaque comparaison.

### Prochaines étapes
1. **Stade pN par patiente** à partir des caractéristiques figées (priorité : c'est l'objectif final).
2. Interprétabilité (cartes d'attention / Grad-CAM sur les faux négatifs) et section éthique.
3. Facultatif : VLM en zéro exemple (PLIP, QuiltNet), HED sur le ResNet18 pour comparer « corriger la couleur » à « modèle robuste ».

### Questions pour l'enseignante
1. Une cohorte réduite de lames complètes (~10 patientes) suffit-elle pour l'étape stade pN ?
2. Un VAE (visualisation du changement de domaine) est-il bienvenu en complément ?
3. Le test final sur le centre 2 doit-il rester intouché jusqu'au rapport, comme prévu ?

### Prochaine étape (semaine 2)
Voir « Priorités semaine 2 » ci-dessus ; en parallèle, téléchargement et découpage des lames complètes.
