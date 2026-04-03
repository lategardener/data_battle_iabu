🏆 Data Battle IA PAU 2026 – Projet …

## 👥 Équipe

**Nom de l’équipe :** IABU

**Membres :**  
- Berthelot Sonny  
- Djolé Marc  
- Vigneron Bastien  
- Buteau Thibault  
- Vanni Marco

## 🎯 Problématique

La problématique est d'arriver à rouvrir un aéroport le plus tôt possible lors d'une alerte d'orage tout en minimisant le risque qu'un éclair frappe l'aéroport.
Les aéroports appliquent généralement une règle fixe : **lever l’alerte 30 minutes après le dernier impact** détecté dans un rayon de **20 km**.  

**Deux métriques officielles (jury) :**
- **Gain G** — nombre total de minutes gagnées par rapport à la baseline humaine des 30 minutes.
- **Risque R** — ratio d’impacts manqués dans la zone d’audit **3 km**. Doit rester sous **R < 2%**.

## 💡 Solution proposée

Nous formulons le problème comme une **classification binaire continue** : à chaque minute d’une alerte en cours, le modèle prédit si l’orage va produire un nouvel impact dangereux de type **CG** dans les **30 prochaines minutes**.

**Choix clés :**
- Grille temporelle à la minute avec features glissantes multi-échelles (5 min, 20 min).
- Suivi géométrique de l’orage : vitesse du centroïde, distance projetée à 30 minutes.
- Ratio IC/CG et tendances de polarité comme signaux de dissipation.
- Règle d’ouverture irrévocable : une fois que l’IA “ouvre” (lève l’alerte), elle ne peut plus la refermer.
- Évaluation sur plusieurs zones de sécurité : **3, 5, 7, 10, 15, 20 km**.

**Modèle :** classifieur XGBoost optimisé via Optuna, entraîné avec GroupKFold (5 folds, split au niveau des orages pour éviter les fuites).

---

## ⚙️ Stack technique

Langages : Python, Jupyter Notebook

Frameworks : Aucun

Outils : Optuna, XGBoost, pandas, NumPy, scikit-learn, joblib, Matplotlib, Seaborn, PyArrow

IA (si utilisée) : XGBoost

## Structure du dépôt

```text
`data_battle/
│
├── config/
│   ├── config.py
│   │   # Fichier central de configuration :
│   │   # - paramètres globaux (zones, seeds, chemins)
│   │   # - hyperparamètres du modèle (XGBoost)
│   │   # - liste des features utilisées
│   └── __init__.py
│       # Permet de traiter config comme un module Python
│
├── data/
│   ├── grid_train.parquet
│   ├── grid_test.parquet
│   │   # Données transformées en grille temporelle (résolution minute)
│   │
│   ├── segment_alerts_all_airports_train.csv
│   └── segment_alerts_all_airports_test.csv
│       # Données brutes initiales (segments d’alertes météo)
│
├── models/
│   └── xgb_final.pkl
│       # Modèle final entraîné (XGBoost sérialisé)
│
├── notebooks/
│   ├── 01_data_processing.ipynb
│   │   # Pipeline de preprocessing :
│   │   # - nettoyage des données
│   │   # - création des features
│   │   # - construction de la grille temporelle
│   │
│   ├── 02_training_and_evaluation.ipynb
│   │   # Entraînement et évaluation :
│   │   # - cross-validation
│   │   # - tuning (Optuna)
│   │   # - métriques métier (risque, gain)
│   │
│   └── code_complete.ipynb
│       # Notebook consolidé :
│       # exécution complète du pipeline de bout en bout
│
├── presentation/
│   ├── data_battle_IABU.pdf
│   │   # Présentation finale du projet (résultats, méthodologie)
│   └── __init__.py
│       # (optionnel) permet import si besoin
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaning.py
│   │   │   # Nettoyage des données :
│   │   │   # - gestion des valeurs aberrantes
│   │   │   # - normalisation / formatage
│   │   │
│   │   ├── features.py
│   │   │   # Feature engineering :
│   │   │   # - extraction de variables explicatives
│   │   │   # - transformations métier
│   │   │
│   │   ├── storm_groups.py
│   │   │   # Regroupement des impacts en événements (orages)
│   │   │   # + génération des variables cibles
│   │   │
│   │   └── __init__.py
│   │
│   ├── temporal_grid/
│   │   ├── grid_builder.py
│   │   │   # Construction d’une grille temporelle (pas = 1 minute)
│   │   │   # + calcul de features glissantes (rolling window)
│   │   │
│   │   └── __init__.py
│   │
│   ├── training/
│   │   ├── model.py
│   │   │   # Entraînement du modèle :
│   │   │   # - cross-validation
│   │   │   # - entraînement final
│   │   │
│   │   ├── tuning.py
│   │   │   # Optimisation des hyperparamètres (Optuna)
│   │   │
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   │   # Calcul des métriques :
│   │   │   # - risque (R)
│   │   │   # - gain (G)
│   │   │   # - scan de seuils
│   │   │
│   │   ├── report.py
│   │   │   # Génération de rapports :
│   │   │   # - synthèse des performances
│   │   │   # - visualisations / exports
│   │   │
│   │   └── __init__.py
│   │
│   ├── pipeline/
│   │   ├── predict.py
│   │   │   # Pipeline d’inférence :
│   │   │   # - prédiction en temps réel (par événement)
│   │   │   # - prédiction batch
│   │   │
│   │   └── __init__.py
│   │
│   └── __init__.py
│       # Rend le dossier src importable comme package
│
├── README.md
│   # Documentation principale du projet :
│   # - description
│   # - installation
│   # - usage
│
├── requirements.txt
│   # Dépendances Python nécessaires
│
└── .gitignore
    # Fichiers ignorés par Git :
    # - environnements virtuels
    # - fichiers lourds (data, modèles)
    # - fichiers IDE (.idea, etc.)
`

## 🚀 Installation & exécution

## Quick start

```bash
# 1. Installer les dépendances
pip install -r requirements.txt


# 2. Lancer les notebooks suivants
notebooks/01_data_processing.ipynb
notebooks/02_training_and_evaluation.ipynb
```

---

## ⚙️ Utilisation du pipeline

### 1. Chargement des données

```python
df_train = pd.read_parquet('data/grid_train.parquet')
df_test  = pd.read_parquet('data/grid_test.parquet')
```

---

### 2. Entraînement (GroupKFold)

```python
oof_preds, fold_models = train_with_cross_validation(
    X_all, y_all, groups_all
)
```

---

### 3. Recherche de seuil

```python
results = find_best_threshold(
    df_train,
    risk_threshold=0.02
)
```

---

### 4. Entraînement final

```python
model = train_final_model(X_all, y_all)
save_model(model, 'models/xgb_final.pkl')
```

---

### 5. Évaluation

```python
results = predict_batch(model, df_test)
```

---

### 6. Inférence temps réel

```python
proba = predict_realtime(model, df_storm)
```
## Data

- 230k éclairs CG (Cloud-Ground) et IC (Intra-Cloud) sur 5 ans, couvrant 10 aéroports français.
- colonnes: `date`, `airport`, `lat`, `lon`, `dist`, `azimuth`, `amplitude`, `icloud`, `maxis`, `is_last_lightning_cloud_ground`.
- L'alerte est déclenchée à la première éclaire CG détectée dans un rayon de 20 km autour de l’aéroport, et se termine 30 minutes après le dernier éclair CG détecté dans ce rayon.
