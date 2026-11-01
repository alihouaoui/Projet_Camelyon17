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

## Semaine 2 (suite) — 01/11/2026 : stade pN par patiente

**Question posée** : à partir du détecteur de patchs, peut-on donner le stade pN de patientes jamais vues ?
Plan critiqué au préalable par 5 relecteurs (faisabilité ITC, taille des tuiles, vérité terrain, statistique, calcul).

**Chaîne de décision** (le stade n'est pas « vu » par un modèle, il est construit comme le fait le pathologiste)
1. **Carte de chaleur par ganglion** (`src/wsi.py`, `scripts/wsi_features.py`) : masque tissu (Otsu sur la saturation),
   toutes les tuiles 96×96 du niveau 2 (~0,97 µm/pixel, ~93 µm par tuile, même échelle que WILDS), lues à distance
   sur le bucket public AWS (requêtes HTTP Range, sans télécharger les lames de ~2 Go) ; Phikon figé + sonde de la semaine 2
   → probabilité de tumeur par tuile.
2. **Classe du ganglion** (négatif / ITC / micro / macro), deux méthodes comparées (`scripts/pn_stage.py`) :
   - **règle de taille** : seuil t, taches d'au moins m tuiles, plus grand diamètre (Feret) en mm, seuils TNM
     (ITC < 0,2 mm ≤ micro ≤ 2 mm < macro, non réglés) ; seuls t et m sont appris ;
   - **forêt aléatoire** sur des résumés de la carte (probabilité max, nombre de tuiles suspectes, diamètres…).
3. **Stade pN** : règles officielles du challenge à partir des 5 ganglions (pas de modèle).

**Données** : `configs/pn_cohort.csv` (`scripts/pn_cohort.py`) — les **57 patientes de CAMELYON17 jamais vues par le
détecteur** (les 43 patientes WILDS sont exclues), 285 lames (205 négatives, 43 macro, 27 micro, 10 ITC).
Vérité : `stages.csv` officiel (taille évaluée par les pathologistes, avec IHC).
Calcul : 285 lames, ~3 millions de tuiles ; Mac (MPS, ~8 lames/h) puis Colab T4 (~1 lame/min).

**Évaluation** : validation croisée par patiente, stratifiée sur le stade, 3 plis × 20 répétitions (3 plis car seulement
3 patientes pN0(i+)) ; métrique officielle : kappa pondéré quadratique.

**Résultats (57 patientes)**

| Méthode (étape 2) | Kappa moyen (min–max sur 20 répétitions) | Stade pN exact | Classe de ganglion exacte |
|---|---|---|---|
| Forêt aléatoire | **0,81** (0,75–0,87) | 70 % | 86 % |
| Règle de taille | 0,79 (0,69–0,83) | 68 % | 87 % |

Matrice de confusion pN (forêt aléatoire, dernière répétition ; lignes = vérité) :

| | pN0 | pN0(i+) | pN1mi | pN1 | pN2 |
|---|---|---|---|---|---|
| pN0 (24) | 20 | 1 | 3 | 0 | 0 |
| pN0(i+) (3) | 2 | 0 | 1 | 0 | 0 |
| pN1mi (9) | 6 | 0 | 2 | 1 | 0 |
| pN1 (14) | 0 | 0 | 0 | 13 | 1 |
| pN2 (7) | 0 | 0 | 0 | 2 | 5 |

Kappa par hôpital (forêt / règle) : C0 0,82 / 0,80 ; C1 0,78 / 0,57 ; C2 0,53 / 0,68 ; C3 0,92 / 0,79 ; C4 0,92 / 0,86
(10 à 13 patientes par hôpital : très instable).
Fichiers : `reports/pn_resume.csv`, `reports/pn_patients.csv`, figure `reports/figures/pn_cartes_exemples.png`
(vrai positif, métastase ratée, fausse alerte).

**Constats**
1. Avec 57 patientes, les deux méthodes se valent (écart de kappa 0,02, dans la variabilité entre répétitions) ;
   la règle de taille a l'avantage d'être explicable (elle reprend la définition médicale).
2. Les cancers avancés sont bien classés (pN1 13/14, pN2 5/7 avec la forêt) ; les erreurs sont surtout d'un cran.
3. **Point faible : les petites métastases** — 6 pN1mi sur 9 classées pN0 ; aucune ITC détectée (plus petites qu'une
   tuile de 93 µm ; la meilleure équipe du challenge n'en trouvait que ~11 %).
4. Repère : meilleure équipe du challenge 0,90 (500 patientes de test) — non comparable directement.

**Limites**
- 57 patientes : chiffres instables (kappa de 0,69 à 0,87 selon le découpage) ; par hôpital, encore plus.
- Les réglages (t, m, forêt) sont choisis dans la validation croisée, pas sur un jeu de test séparé.
- La vérité « taille » des pathologistes vient de l'IHC, que le modèle ne voit pas (H&E seulement).
- Incidents traités : lames Philips sans balise de résolution (lue dans la description XML), une lame à 7 niveaux,
  mémoire GPU saturée sur les très grosses lames (réduction automatique du lot).

## Semaine 2 (suite) — 01/11/2026 : ABMIL pour l'étape 2

**Question posée** : un modèle multi-instances appris sur les caractéristiques Phikon fait-il mieux que la règle et la forêt ?
Plan critiqué par 5 relecteurs (statistique, architecture, clinique, calcul, valeur pour le projet) puis **figé avant lancement**.

**Constat clé des relecteurs** : l'attention fait une moyenne pondérée des tuiles, elle ne mesure ni taille ni forme ;
elle ne peut donc pas, en principe, trancher micro (0,2–2 mm) contre macro (> 2 mm). D'où :
- **Principal — ABMIL hybride** : ABMIL binaire (lame atteinte micro/macro ou non ; ITC compté négatif) décide
  « atteinte ou pas » ; la taille vient de la carte de la sonde (règle apprise sur les plis d'entraînement) ;
- **Secondaire — ABMIL 3 classes** (négative + ITC / micro / macro) direct, pour voir ce qu'il ne sait pas faire.

**Méthode** (`scripts/pn_abmil.py`) : attention à porte (Ilse et al., 2018) réduite (768→128, ~110 k paramètres) ;
entrée Phikon seule (ni coordonnées ni sonde) ; AdamW, 20 époques fixes, une graine, pas d'arrêt anticipé ;
entraînement sur toutes les tuiles suspectes (sonde ≥ 0,3) + tirage jusqu'à 8 000 tuiles, inférence sur toutes ;
**mêmes plis** que la règle et la forêt (recalculées dans la même boucle), 3 plis × 10 répétitions ;
écart de kappa apparié par répétition, IC 95 % par bootstrap des patientes (1 000 tirages).
Calcul : CPU du Mac, ~20 min par répétition (6 entraînements), 3 h 30 au total.

**Résultats (57 patientes)**

| Méthode | Kappa moyen (min–max) | pN exact | Métastatiques ratées* | Macro ratées* | Sur-stadification* |
|---|---|---|---|---|---|
| **ABMIL hybride** | **0,89** (0,86–0,93) | **79 %** | **4,4** | 2,1 | 1,5 |
| ABMIL 3 classes | 0,87 (0,85–0,91) | 78 % | 5,1 | 1,3 | **1,2** |
| Forêt aléatoire | 0,80 (0,76–0,87) | 69 % | 6,3 | **0,5** | 4,4 |
| Règle de taille | 0,80 (0,77–0,83) | 69 % | 6,4 | 2,1 | 5,4 |

\* moyenne sur les 10 répétitions, en nombre de patientes sur 57 ; « métastatiques ratées » = pN1mi/pN1/pN2 classées
pN0 ou pN0(i+) ; « macro ratées » = pN1/pN2 classées en dessous de pN1 ; « sur-stadification » = pN0/pN0(i+) classées métastatiques.

Écart de kappa [IC 95 %] : ABMIL hybride − règle **+0,09 [0,01 ; 0,18]** ; ABMIL hybride − forêt **+0,08 [0,01 ; 0,18]** ;
ABMIL 3 classes − règle +0,07 [−0,00 ; 0,16] ; ABMIL 3 classes − forêt +0,07 [−0,00 ; 0,17].
Fichiers : `reports/pn_abmil_resume.csv`, `reports/pn_abmil_ecarts.csv`.

**Constats**
1. L'ABMIL hybride est meilleur que la règle et la forêt **dans les 10 répétitions** ; l'IC de l'écart exclut 0,
   mais sa borne basse est proche de 0 (57 patientes).
2. Le gain vient de la **détection des petites métastases** (4,4 patientes métastatiques ratées au lieu de ~6,4) et de
   **3 fois moins de fausses alertes**.
3. Compromis : la forêt rate moins de macro-métastases (0,5 contre 2,1), l'erreur la plus grave cliniquement.
4. Contrairement à l'attente des relecteurs, l'ABMIL 3 classes fait presque aussi bien (0,87) ; écart non démontré.
   Le modèle principal reste l'hybride, choisi avant lancement.
5. Toujours aucune ITC détectée ; 4 à 5 pN1mi restent classées pN0.

**Limites** : une seule graine par pli ; hyperparamètres non réglés (choix volontaire) ; même cohorte de 57 patientes
pour toutes les comparaisons (pas de test externe) ; l'attention (`outputs/pn_abmil/attention.npz`) n'est pas
une explication causale.

### Prochaines étapes
1. Interprétabilité : cartes d'attention contre cartes de la sonde sur les pN1mi ratées ; section éthique
   (outil d'aide à la relecture, coût asymétrique des erreurs, relecture humaine des lames négatives).
2. Facultatif : VLM de pathologie (CONCH) comme autre détecteur à l'étape 1.
3. Rapport (15 pages) et soutenance (10 min).

### Questions pour l'enseignante
1. Un VAE (visualisation du changement de domaine) est-il bienvenu en complément ?
2. Le test final sur le centre 2 doit-il rester intouché jusqu'au rapport, comme prévu ?
