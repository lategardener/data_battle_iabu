"""
Pipeline d’inférence pour la prédiction de fin d’alerte orage.

Deux modes :
- real_time : un seul orage, retourne la probabilité brute pour la minute courante.
- test : plusieurs orages, retourne les statistiques de gain/risque par orage pour chaque
  zone de sécurité demandée.
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config.config import (
    FEATURE_COLUMNS,
    RISK_THRESHOLD,
    SAFETY_ZONES_KM,
    SMOOTHING_WINDOW,
    CONFIRMATION_WINDOW,
    HORIZON_MIN,
)
from src.evaluation.metrics import (
    apply_irrevocable_decision,
    compute_gain,
    compute_risk,
    find_best_threshold,
)
from src.temporal_grid.grid_builder import build_temporal_grid


def load_model(model_path: str) -> xgb.XGBClassifier:
    """Charge un modèle XGBoost entraîné depuis le disque."""
    model = joblib.load(model_path)
    print(f"Modèle chargé depuis {model_path}")
    return model


def save_model(model: xgb.XGBClassifier, model_path: str) -> None:
    """Sauvegarde un modèle entraîné sur le disque."""
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")


def _prepare_grid(
    df_raw: pd.DataFrame,
    safety_zones_km: list[int] = SAFETY_ZONES_KM,
) -> pd.DataFrame:
    """
    Construit une grille à résolution minute à partir des données brutes de foudre.
    Bypass automatique si les données sont déjà formatées en grille.
    """
    # Bypass : si count_cg_20km est présent, c'est que la grille est déjà faite
    if "count_cg_20km" in df_raw.columns:
        print("Grille temporelle détectée. Passage direct à l'inférence...")
        return df_raw.copy()

    print("Données brutes détectées. Construction de la grille temporelle...")
    return build_temporal_grid(
        df_raw,
        horizon_min=HORIZON_MIN,
        safety_zones_km=safety_zones_km,
    )


def _predict_proba(
    model: xgb.XGBClassifier,
    df_grid: pd.DataFrame,
    feature_cols: list[str],
    smoothing_window: int = SMOOTHING_WINDOW,
) -> pd.DataFrame:
    """Ajoute les colonnes de probabilités brute et lissée à la grille."""
    cols = [c for c in feature_cols if c in df_grid.columns]

    if len(cols) == 0:
        raise ValueError("Aucune feature trouvée dans le DataFrame pour faire la prédiction.")

    df_grid["proba_brute"] = model.predict_proba(df_grid[cols])[:, 1]

    # Lissage par orage
    df_grid["proba_lissee"] = df_grid.groupby("storm_group_id")["proba_brute"].transform(
        lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
    )
    return df_grid


def predict_realtime(
    model: xgb.XGBClassifier,
    df_current_storm: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    safety_zones_km: list[int] = SAFETY_ZONES_KM,
) -> float:
    """
    Inférence en temps réel pour un seul orage en cours.
    """
    df_grid = _prepare_grid(df_current_storm, safety_zones_km=safety_zones_km)
    df_grid = _predict_proba(model, df_grid, feature_cols)

    latest_proba = df_grid["proba_lissee"].iloc[-1]
    print(f"Probabilité actuelle de danger : {latest_proba:.4f}")
    return float(latest_proba)


def predict_batch(
    model: xgb.XGBClassifier,
    df_data: pd.DataFrame,
    safety_zones_km: list[int] = SAFETY_ZONES_KM,
    feature_cols: list[str] = FEATURE_COLUMNS,
    risk_threshold: float = RISK_THRESHOLD,
    find_threshold: bool = True,
    threshold_override: float | None = None,
) -> dict:
    """
    Inférence batch pour plusieurs orages.
    Supporte l'évaluation multi-zones avec audit dynamique.
    """
    # 1. Grille et Prédictions
    df_grid = _prepare_grid(df_data, safety_zones_km=safety_zones_km)
    df_grid = _predict_proba(model, df_grid, feature_cols)

    results: dict = {}

    if threshold_override is not None:
        # 2a. Évaluation avec un seuil forcé
        print(f"Application du seuil forcé : {threshold_override:.4f}")
        for zone_km in safety_zones_km:
            audit_col = f"count_cg_{zone_km}km"

            decision = apply_irrevocable_decision(
                df_grid,
                threshold=threshold_override,
                proba_col="proba_lissee",
                alert_zone_col="count_cg_20km",
                confirmation_window=CONFIRMATION_WINDOW,
            )

            gains = compute_gain(df_grid, decision, alert_zone_col="count_cg_20km")
            risk = compute_risk(df_grid, decision, audit_col=audit_col)

            results[zone_km] = _build_zone_result(
                threshold_override, gains, risk, zone_km
            )

    elif find_threshold:
        # 2b. Recherche automatique du meilleur seuil
        results = find_best_threshold(
            df_grid,
            safety_zones_km=safety_zones_km,
            proba_col="proba_lissee",
            risk_threshold=risk_threshold,
            confirmation_window=CONFIRMATION_WINDOW,
            horizon_min=HORIZON_MIN,
        )

        # Ajout détaillé des gains par orage pour l'analyse
        for zone_km, res in results.items():
            if res.get("total_gain", -1) >= 0:
                decision = apply_irrevocable_decision(
                    df_grid,
                    threshold=res["threshold"],
                    proba_col="proba_lissee",
                    alert_zone_col="count_cg_20km",
                    confirmation_window=CONFIRMATION_WINDOW,
                )
                res["gains_per_storm"] = compute_gain(df_grid, decision, alert_zone_col="count_cg_20km")

    _print_batch_summary(results, safety_zones_km)
    return results


def _build_zone_result(
    threshold: float,
    gains: pd.Series,
    risk: float,
    zone_km: int,
) -> dict:
    """Construit un dictionnaire de résultat standardisé pour une zone."""
    return {
        "threshold": threshold,
        "total_gain": int(gains.sum()),
        "risk": risk,
        "stats": {
            "mean": gains.mean(),
            "median": gains.median(),
            "std": gains.std(),
            "max": gains.max(),
            "min": gains.min(),
        },
        "gains_per_storm": gains,
    }


def _print_batch_summary(results: dict, safety_zones_km: list[int]) -> None:
    """Affiche un tableau récapitulatif concis pour toutes les zones."""
    print("\n" + "=" * 75)
    print(f"{'Zone (km)':<12}{'Seuil':>12}{'Gain (min)':>14}{'Risque (%)':>12}")
    print("-" * 75)
    for zone_km in safety_zones_km:
        r = results.get(zone_km, {})
        if r.get("total_gain", -1) >= 0:
            print(
                f"{zone_km:<12}{r['threshold']*100:>11.3f}%"
                f"{r['total_gain']:>14}{r['risk']*100:>11.3f}%"
            )
        else:
            print(f"{zone_km:<12}{'—':>12}{'aucun seuil valide':>26}")
    print("=" * 75)