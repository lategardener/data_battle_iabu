"""
Recherche d'hyperparamètres Optuna pour XGBoost.
Deux objectifs sont fournis :
- maximize_gain : maximiser le gain de temps avec zéro crash partout
- maximize_gain_3km : maximiser le gain avec zéro crash fatal (<3 km)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold

from config.config import CONFIRMATION_WINDOW, SMOOTHING_WINDOW


def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: dict,
    n_splits: int = 5,
) -> np.ndarray:
    """Exécute la validation croisée GroupKFold et retourne les prédictions OOF."""
    oof = np.zeros(len(X))
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


def _smooth(df_eval: pd.DataFrame, oof: np.ndarray) -> pd.Series:
    """Attache les probabilités OOF et applique un lissage par orage."""
    df_eval = df_eval.copy()
    # On prend uniquement les indices présents dans df_eval
    df_eval["proba_brute"] = oof[df_eval.index]
    return df_eval.groupby("storm_group_id")["proba_brute"].transform(
        lambda x: x.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    )


def build_objective_maximize_gain(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    df_eval: pd.DataFrame,
) -> callable:
    """
    Objectif Optuna : maximiser le gain de temps avec zéro crash n'importe où dans la zone de danger.

    Arguments :
        X : Matrice des caractéristiques.
        y : Vecteur cible.
        groups : Identifiants des groupes d'orages.
        df_eval : DataFrame d'évaluation avec les colonnes masque_danger et masque_gain.

    Retourne :
        Fonction objectif Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "gamma": trial.suggest_float("gamma", 1.0, 4.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 6.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        oof = _run_cv(X, y, groups, params)
        df_tmp = df_eval.copy()
        df_tmp["proba_lissee"] = _smooth(df_tmp, oof)

        best_gain = -1000
        for seuil in np.linspace(0.001, 0.15, 30):
            decision = (df_tmp["proba_lissee"] > seuil).astype(int)
            decision_ia = df_tmp.groupby("storm_group_id")["proba_lissee"].transform(
                lambda x: (x > seuil).rolling(
                    window=CONFIRMATION_WINDOW, min_periods=1
                ).max()
            )
            crashs = (df_tmp.loc[df_tmp["masque_danger"], decision_ia.name] == 0).sum()
            if crashs == 0:
                gain = (df_tmp.loc[df_tmp["masque_gain"], decision_ia.name] == 0).sum()
                best_gain = max(best_gain, gain)

        return best_gain

    return objective


def build_objective_maximize_gain_distance(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    df_eval: pd.DataFrame,
    safety_distance_km: float = 3.0,  # distance paramétrable
    max_threshold: float = 0.3
) -> callable:
    """
    Objectif Optuna : maximiser le gain de temps avec zéro événement fatal
    dans la zone de danger, pour n'importe quelle distance de sécurité.

    Arguments :
        X : DataFrame des features.
        y : Série cible.
        groups : IDs de groupe (storm_group_id).
        df_eval : DataFrame d'évaluation avec masque_danger, masque_gain, count_cg_audit.
        safety_distance_km : distance de sécurité pour définir les crashs fatals.
        max_threshold : seuil max à tester pour la décision IA.

    Retourne :
        Fonction objectif Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "gamma": trial.suggest_float("gamma", 1.0, 4.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 6.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        # 1. Prédictions OOF
        oof = _run_cv(X, y, groups, params)
        df_tmp = df_eval.copy()
        df_tmp["proba_lissee"] = _smooth(df_tmp, oof)

        best_gain = -1000
        seuils = np.linspace(0.001, max_threshold, 40)

        for seuil in seuils:
            # Décision brute + décision confirmée
            df_tmp["decision_brute"] = (df_tmp["proba_lissee"] > seuil).astype(int)
            df_tmp["decision_ia"] = df_tmp.groupby("storm_group_id")["decision_brute"].transform(
                lambda x: x.rolling(window=CONFIRMATION_WINDOW, min_periods=1).max()
            )

            # Calcul du gain
            gain = (df_tmp.loc[df_tmp["masque_gain"], "decision_ia"] == 0).sum()

            # Vérification de fatalité dans la distance de sécurité
            fatals = 0
            zone_danger = df_tmp[df_tmp["masque_danger"]]
            ouvertures = zone_danger[df_tmp.loc[zone_danger.index, "decision_ia"] == 0]

            if not ouvertures.empty:
                for storm_id in ouvertures["storm_group_id"].unique():
                    df_s = df_tmp[
                        (df_tmp["storm_group_id"] == storm_id) & df_tmp["masque_danger"]
                    ]
                    ouv = df_s[df_s["decision_ia"] == 0]
                    if not ouv.empty:
                        zone_post = df_s.loc[ouv.index[0]:]
                        hits = zone_post[zone_post["count_cg_audit"] > 0]
                        if not hits.empty and hits["last_known_dist"].min() < safety_distance_km:
                            fatals += 1
                            break

            if fatals == 0:
                best_gain = max(best_gain, gain)

        return best_gain

    return objective


def run_optuna(
    objective: callable,
    n_trials: int = 30,
    best_known_params: dict | None = None,
) -> optuna.Study:
    """
    Exécute une étude Optuna.

    Arguments :
        objective : Fonction objectif Optuna.
        n_trials : Nombre d'essais à exécuter.
        best_known_params : Si fourni, met en file d'attente comme premier essai.

    Retourne :
        Étude Optuna terminée.
    """
    study = optuna.create_study(direction="maximize")

    if best_known_params is not None:
        study.enqueue_trial(best_known_params)

    study.optimize(objective, n_trials=n_trials)

    print("\nMeilleur essai :")
    print(f"  Gain : {study.best_value}")
    for k, v in study.best_params.items():
        print(f"  {k} : {v}")

    return study