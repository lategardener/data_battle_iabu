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
data_battle/
│
├── config/
│   └── config.py            # paramètres : zones, hyperparamètres XGB, liste des features
│
├── src/
│   ├── preprocessing/
│   │   ├── storm_groups.py  # regroupement des impacts en orages + calcul de cibles
│   │   ├── cleaning.py      # nettoyage : anomalies, bruit, formatage
│   │   └── features.py      # feature engineering sur impacts bruts
│   │
│   ├── temporal_grid/
│   │   └── grid_builder.py  # grille à 1 minute + features glissantes
│   │
│   ├── training/
│   │   ├── model.py         # entraînement CV et entraînement final
│   │   └── tuning.py        # objectifs Optuna
│   │
│   ├── evaluation/
│   │   └── metrics.py       # risque R, gain G, scan de seuils, rapports
│   │
│   └── pipeline/
│       └── predict.py       # inférence : temps réel (un orage) et batch
│
├── notebooks/
│   ├── 01_data_processing.ipynb          # données brutes → grille temporelle
│   └── 02_training_and_evaluation.ipynb  # entraînement, tuning, évaluation, plots
│
├── models/                  # modèles sauvegardés (.pkl) — non commités
├── data/                    # données brutes et transformées — non commitées
├── requirements.txt
└── .gitignore


## 🚀 Installation & exécution

## Quick start

```bash
# 1. Installer les dépendances
pip install -r requirements.txt


# 2. Lancer les notebooks dans l’ordre
jupyter notebook notebooks/01_data_processing.ipynb
jupyter notebook notebooks/02_training_and_evaluation.ipynb
```

---

## Utilisation de le pipeline

### Entraînement du modèle final

```python
from src.pipeline.predict import predict_batch, save_model
from src.training.model import train_final_model
from config.config import FEATURE_COLUMNS

model = train_final_model(X_all, y_all)
save_model(model, 'models/xgb_final.pkl')
```

### Batch inference (évaluation sur un jeu de test)

```python
from src.pipeline.predict import load_model, predict_batch

model = load_model('models/xgb_final.pkl')

results = predict_batch(
    model,
    df_test,                       # données de test prétraitées en grille temporelle
    safety_zones_km=[3, 5, 10],    # zones d’audit à évaluer
    risk_threshold=0.02,
    find_threshold=True,           # optimise le seuil de décision pour R < 2% sur la zone d’audit 3 km
)

# Affichage des résultats pour la zone d’audit 3 km
print(results[3]['total_gain'])     # total minutes gagnées sur la zone d’audit 3 km
print(results[3]['risk'])           # risque R à 3 km
print(results[3]['stats'])          # détails des statistiques
```

### Vraie inférence temps réel (un seul orage, minute par minute)

```python
from src.pipeline.predict import load_model, predict_realtime

model = load_model('models/xgb_final.pkl')

#  df_storm: historique brut des éclairs pour l’orage actuel (aéroport unique)
proba = predict_realtime(model, df_storm)
print(f"Probabilité de vrai danger: {proba:.4f}")
```

---

## Modèle et hyperparamètres (cas de la zone d’audit 3 km)

Obtenu avec optuna optimisant le gain G sous la contrainte R < 2% sur la zone d’audit 3 km.:

```python
XGB_DEFAULT_PARAMS = {
    "n_estimators": 506,
    "max_depth": 5,
    "learning_rate": 0.10423180025726148,
    "min_child_weight": 10,
    "subsample": 0.8738358835624531,
    "colsample_bytree": 0.8723751225535342,
    "gamma": 1.7503741438290679,
    "scale_pos_weight": 3.964987197971567,
}
```

Toutes les features utilisées sont listées dans `config/config.py`

---

## Changez la zone d’audit

La zone de sécurité par défaut est de 20 km, mais vous pouvez évaluer le modèle sur plusieurs zones d’audit en même temps (3, 5, 7, 10, 15, 20 km) pour analyser le compromis gain/risque à différentes distances.
```python
# Evaluation de plusieurs zones d’audit : 3, 5, 7, 10, 15, 20 km
results = predict_batch(model, df_raw, safety_zones_km=[3, 5, 7, 10, 15, 20])
```

The alert trigger zone (20 km by default) is set by `ALERT_ZONE_KM` in `config/config.py`.

---

## Data

- 230k éclaises CG (Cloud-Ground) et IC (Intra-Cloud) sur 5 ans, couvrant 10 aéroports français.
- colonnes: `date`, `airport`, `lat`, `lon`, `dist`, `azimuth`, `amplitude`, `icloud`, `maxis`, `is_last_lightning_cloud_ground`.
- L'alerte est déclenchée à la première éclaire CG détectée dans un rayon de 20 km autour de l’aéroport, et se termine 30 minutes après la dernière éclaire CG détectée dans ce rayon.
