"""
Configuration globale du pipeline de prédiction de fin d’alerte orage.
Tous les seuils et hyperparamètres du modèle sont définis ici.
"""

# ──────────────────────────────────────────────
# Paramètres de zone d’alerte
# ──────────────────────────────────────────────

# Rayon (km) définissant la zone d’alerte autour d’un aéroport
ALERT_ZONE_KM: int = 20

# Rayons de sécurité (km) utilisés pour l’évaluation multi-échelles du risque
SAFETY_ZONES_KM: list[int] = [3, 5, 7, 10, 15, 20]

# Zone de sécurité par défaut utilisée pour le calcul du risque (métrique jury)
DEFAULT_RISK_KM: int = 3

# Risque maximal acceptable (ratio d’impacts de foudre manqués)
RISK_THRESHOLD: float = 0.02

# ──────────────────────────────────────────────
# Paramètres de feature engineering
# ──────────────────────────────────────────────

ROLLING_WINDOW: int = 10       # Nombre d’impacts pour les features glissantes
CONTEXT_SIZE: int = 20         # Nombre d’impacts de contexte avant le début d’alerte
CONTEXT_WINDOW_MIN: int = 60   # Fenêtre temporelle (min) pour inclure le contexte
HORIZON_MIN: int = 30          # Horizon de prédiction (baseline)

# ──────────────────────────────────────────────
# Paramètres du modèle (meilleur essai Optuna)
# ──────────────────────────────────────────────

XGB_DEFAULT_PARAMS: dict = {
    "n_estimators": 506,
    "max_depth": 5,
    "learning_rate": 0.10423180025726148,
    "min_child_weight": 10,
    "subsample": 0.8738358835624531,
    "colsample_bytree": 0.8723751225535342,
    "gamma": 1.7503741438290679,
    "scale_pos_weight": 3.964987197971567,
    "random_state": 42,
    "n_jobs": -1,
}

# ──────────────────────────────────────────────
# Paramètres d’entraînement
# ──────────────────────────────────────────────

N_CV_SPLITS: int = 5           # Nombre de splits GroupKFold
SMOOTHING_WINDOW: int = 3      # Fenêtre glissante pour lisser les probabilités
CONFIRMATION_WINDOW: int = 10  # Minutes nécessaires pour confirmer une décision d’ouverture

# ──────────────────────────────────────────────
# Encodage des aéroports
# ──────────────────────────────────────────────

AIRPORT_MAPPING: dict[str, int] = {
    "Bastia": 1,
    "Ajaccio": 1,
    "Nantes": 2,
    "Pise": 3,
    "Biarritz": 4,
}

# ──────────────────────────────────────────────
# Liste des features utilisées pour l’entraînement
# ──────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # Chronologie et maturité de l’orage
    "minutes_since_last_strike",
    "cum_n_strikes",
    # Court terme (5 min)
    "activity_count_last_5m",
    "dist_last_5m",
    "delta_dist_last_5m",
    "abs_amplitude_last_5m",
    # Moyen terme (20 min)
    "activity_count_last_20m",
    "dist_last_20m",
    "distance_macro_trend",
    "activity_drop_ratio",
    # Géométrie et projections
    "vitesse_eloignement",
    "dist_projetee_30m",
    "dist_centroid_actuel",
    "dist_centroid_proj_30m",
    "centroid_eloignement_net",
    # Physique de l’orage
    "count_ic_last_20m",
    "count_cg_last_20m",
    "count_pos_last_20m",
    "ratio_ic_cg_last_20m",
    "ratio_pos_last_20m",
    # Contexte temporel
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Features avancées
    "azimuth_std_last_5m",
    "storm_spread_radial",
    "energie_last_5m",
    "energy_drop_ratio",
    # Mémoire (distance sans impact récent)
    "last_known_dist",
]