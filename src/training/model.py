"""
Entraînement du modèle avec validation croisée GroupKFold.
Les paramètres par défaut de XGBoost proviennent du meilleur essai Optuna.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from config.config import XGB_DEFAULT_PARAMS, N_CV_SPLITS, SMOOTHING_WINDOW


def get_default_model() -> xgb.XGBClassifier:
    """Retourne un XGBClassifier avec les meilleurs hyperparamètres optimisés."""
    return xgb.XGBClassifier(**XGB_DEFAULT_PARAMS)


def train_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: dict | None = None,
    n_splits: int = N_CV_SPLITS,
) -> tuple[np.ndarray, list[xgb.XGBClassifier]]:
    """
    Entraîne XGBoost en utilisant la validation croisée GroupKFold.

    Chaque groupe d'orages apparaît dans exactement un pli de validation, empêchant
    toute fuite de données entre les orages.

    Arguments :
        X : Matrice des caractéristiques (features).
        y : Cible binaire (danger dans les 30 prochaines minutes).
        groups : Identifiants des groupes d'orages pour la séparation des plis.
        params : Paramètres XGBoost. Par défaut XGB_DEFAULT_PARAMS.
        n_splits : Nombre de plis de validation croisée.

    Retourne :
        oof_predictions : Prédictions de probabilité hors pli (out-of-fold) pour l'ensemble de données complet.
        fold_models : Liste des modèles entraînés pour chaque pli.
    """
    if params is None:
        params = XGB_DEFAULT_PARAMS

    oof_preds = np.zeros(len(X))
    fold_models: list[xgb.XGBClassifier] = []
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"  Pli {fold}/{n_splits}...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_models.append(model)

    print("Validation croisée terminée.")
    return oof_preds, fold_models


def smooth_probabilities(
    df: pd.DataFrame,
    raw_proba_col: str = "proba_brute",
    window: int = SMOOTHING_WINDOW,
) -> pd.Series:
    """
    Applique une moyenne mobile par orage pour lisser les probabilités brutes.

    Arguments :
        df : DataFrame avec storm_group_id et la colonne de probabilité brute.
        raw_proba_col : Nom de la colonne des probabilités brutes.
        window : Taille de la fenêtre glissante (minutes).

    Retourne :
        Série de probabilités lissées alignée avec df.
    """
    return df.groupby("storm_group_id")[raw_proba_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """
    Entraîne sur l'ensemble de données complet (utilisé après la validation croisée pour la soumission).

    Arguments :
        X : Matrice complète des caractéristiques.
        y : Vecteur cible complet.
        params : Paramètres XGBoost. Par défaut XGB_DEFAULT_PARAMS.

    Retourne :
        XGBClassifier entraîné.
    """
    if params is None:
        params = XGB_DEFAULT_PARAMS

    model = xgb.XGBClassifier(**params)
    print("Entraînement du modèle final sur 100% des données...")
    model.fit(X, y)
    print("Modèle final prêt.")
    return model