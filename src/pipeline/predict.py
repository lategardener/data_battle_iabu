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
    ALERT_ZONE_KM,
    FEATURE_COLUMNS,
    RISK_THRESHOLD,
    SAFETY_ZONES_KM,
    SMOOTHING_WINDOW,
    CONFIRMATION_WINDOW,
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
    alert_zone_km: int = ALERT_ZONE_KM,
    audit_zone_km: int = 3,
) -> pd.DataFrame:
    """Construit une grille à résolution minute à partir des données brutes de foudre."""
    return build_temporal_grid(
        df_raw,
        horizon_min=30,
        alert_zone_km=float(alert_zone_km),
        audit_zone_km=float(audit_zone_km),
    )


def _predict_proba(
    model: xgb.XGBClassifier,
    df_grid: pd.DataFrame,
    feature_cols: list[str],
    smoothing_window: int = SMOOTHING_WINDOW,
) -> pd.DataFrame:
    """Ajoute les colonnes de probabilités brute et lissée à la grille."""
    cols = [c for c in feature_cols if c in df_grid.columns]
    df_grid["proba_brute"] = model.predict_proba(df_grid[cols])[:, 1]
    df_grid["proba_lissee"] = df_grid.groupby("storm_group_id")["proba_brute"].transform(
        lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
    )
    return df_grid


def predict_realtime(
    model: xgb.XGBClassifier,
    df_current_storm: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    alert_zone_km: int = ALERT_ZONE_KM,
) -> float:
    """
    Inférence en temps réel pour un seul orage en cours.

    Construit la grille temporelle minute à partir de l’historique fourni
    et retourne la probabilité lissée de danger pour la minute la plus récente.

    Args:
        model: XGBClassifier entraîné.
        df_current_storm: Données brutes de foudre pour l’orage courant (un seul orage).
        feature_cols: Features utilisées pour la prédiction.
        alert_zone_km: Rayon de la zone d’alerte.

    Returns:
        Probabilité lissée que l’orage soit encore dangereux (entre 0 et 1).
    """
    df_grid = _prepare_grid(df_current_storm, alert_zone_km=alert_zone_km)
    df_grid = _predict_proba(model, df_grid, feature_cols)

    latest_proba = df_grid["proba_lissee"].iloc[-1]
    print(f"Probabilité actuelle de danger : {latest_proba:.4f}")
    return float(latest_proba)


def predict_batch(
    model: xgb.XGBClassifier,
    df_raw: pd.DataFrame,
    safety_zones_km: list[int] = SAFETY_ZONES_KM,
    feature_cols: list[str] = FEATURE_COLUMNS,
    alert_zone_km: int = ALERT_ZONE_KM,
    risk_threshold: float = RISK_THRESHOLD,
    find_threshold: bool = True,
    threshold_override: float | None = None,
) -> dict:
    """
    Inférence batch pour plusieurs orages.

    Calcule les métriques de gain et de risque pour chaque zone de sécurité demandée.
    Peut soit chercher le seuil optimal, soit utiliser un seuil fixe.

    Args:
        model: XGBClassifier entraîné.
        df_raw: Données brutes contenant plusieurs orages.
        safety_zones_km: Liste des rayons d’audit (km) à évaluer.
        feature_cols: Features utilisées pour la prédiction.
        alert_zone_km: Rayon de la zone d’alerte utilisé au preprocessing.
        risk_threshold: Risque maximal acceptable (R_accept).
        find_threshold: Si True, recherche le meilleur seuil par zone.
        threshold_override: Seuil fixe utilisé pour toutes les zones (ignore la recherche).

    Returns:
        results: Dict[zone_km -> Dict] avec :
            - threshold: float
            - total_gain: int (minutes)
            - risk: float
            - stats: dict (mean, median, std, max, min par orage)
            - gains_per_storm: pd.Series
    """
    df_grid = _prepare_grid(df_raw, alert_zone_km=alert_zone_km)
    df_grid = _predict_proba(model, df_grid, feature_cols)

    results: dict = {}

    if threshold_override is not None:
        # Seuil fixe appliqué à toutes les zones
        for zone_km in safety_zones_km:
            decision = apply_irrevocable_decision(
                df_grid,
                threshold=threshold_override,
                confirmation_window=CONFIRMATION_WINDOW,
            )
            gains = compute_gain(df_grid, decision)
            risk = compute_risk(df_grid, decision)
            results[zone_km] = _build_zone_result(
                threshold_override, gains, risk, zone_km
            )

    elif find_threshold:
        # Recherche du meilleur seuil par zone
        results = find_best_threshold(
            df_grid,
            safety_zones_km=safety_zones_km,
            risk_threshold=risk_threshold,
            confirmation_window=CONFIRMATION_WINDOW,
        )
        # Ajout des gains par orage pour chaque zone
        for zone_km, res in results.items():
            if res["total_gain"] >= 0:
                decision = apply_irrevocable_decision(
                    df_grid,
                    threshold=res["threshold"],
                    confirmation_window=CONFIRMATION_WINDOW,
                )
                res["gains_per_storm"] = compute_gain(df_grid, decision)

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