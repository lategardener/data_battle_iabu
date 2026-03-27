"""
Métriques d’évaluation pour le défi de prédiction de fin d’alerte orage.
Support multi-zones dynamique (3km, 5km, 7km, 10km, 15km, 20km, etc.)
"""

import numpy as np
import pandas as pd


def compute_alert_masks(
    df: pd.DataFrame,
    horizon_min: int = 30,
) -> pd.DataFrame:
    """
    Calcule masque_danger et masque_gain pour chaque orage.
    """
    df = df.copy()
    df["masque_danger"] = False
    df["masque_gain"] = False

    for storm_id, df_storm in df.groupby("storm_group_id"):
        if "cible_ia" not in df_storm.columns or df_storm["cible_ia"].sum() == 0:
            continue
        idx_last = df_storm["cible_ia"][::-1].idxmax()
        fin_alerte = idx_last + horizon_min

        df.loc[df_storm.loc[:idx_last].index, "masque_danger"] = True
        df.loc[df_storm.loc[idx_last:fin_alerte].iloc[1:].index, "masque_gain"] = True

    return df


def apply_irrevocable_decision(
    df: pd.DataFrame,
    threshold: float,
    proba_col: str = "proba_lissee",
    alert_zone_col: str = "count_cg_20km",
    confirmation_window: int = 10,
) -> pd.Series:
    """
    Applique la règle d’ouverture irrévocable.
    """
    grp = df["storm_group_id"]

    # Décision brute par IA
    decision_brute = (df[proba_col] > threshold).astype(int)
    decision_ia = decision_brute.groupby(grp).transform(
        lambda x: x.rolling(window=confirmation_window, min_periods=1).max()
    )

    # Déclenchement de l’alerte humaine : impact CG dans la zone d'alerte
    if alert_zone_col not in df.columns:
        raise ValueError(f"Colonne {alert_zone_col} introuvable dans le DataFrame.")

    eclair_critique = (df[alert_zone_col] > 0).astype(int)
    alerte_declenchee = eclair_critique.groupby(grp).cummax()

    alerte_prec = alerte_declenchee.groupby(grp).shift(1).fillna(0)
    ordre_ouverture = (alerte_prec == 1) & (decision_ia == 0)
    est_ouvert = ordre_ouverture.groupby(grp).cummax()

    decision_irrevocable = ((alerte_declenchee == 1) & (~est_ouvert)).astype(int)
    return decision_irrevocable


def compute_gain(
    df: pd.DataFrame,
    decision_irrevocable: pd.Series,
    alert_zone_col: str = "count_cg_20km",
) -> pd.Series:
    """
    Calcule les minutes gagnées par rapport à la règle humaine de 30 minutes.
    """
    eclair_critique = (df[alert_zone_col] > 0).astype(int)
    alerte_declenchee = eclair_critique.groupby(df["storm_group_id"]).cummax()

    etat_humain = eclair_critique.groupby(df["storm_group_id"]).transform(
        lambda x: x.rolling(window=30, min_periods=1).max()
    )

    minutes_gagnees = (
        (etat_humain == 1) & (decision_irrevocable == 0) & (alerte_declenchee == 1)
    )
    return minutes_gagnees.groupby(df["storm_group_id"]).sum()


def compute_risk(
    df: pd.DataFrame,
    decision_irrevocable: pd.Series,
    alert_zone_col: str = "count_cg_20km",
    audit_col: str = "count_cg_3km",
) -> float:
    """
    Calcule le risque R = M_L3 / N_L3 (impacts dangereux manqués / total des impacts dangereux).
    """
    if audit_col not in df.columns:
        raise ValueError(f"Colonne d'audit {audit_col} introuvable.")

    N_L3 = df[audit_col].sum()
    if N_L3 == 0:
        return 0.0

    eclair_critique = (df[alert_zone_col] > 0).astype(int)
    alerte_declenchee = eclair_critique.groupby(df["storm_group_id"]).cummax()

    ouvertures = (decision_irrevocable == 0) & (alerte_declenchee == 1)
    M_L3 = df.loc[ouvertures, audit_col].sum()

    return float(M_L3 / N_L3)



ALERT_COL = "count_cg_20km"  # déclencheur toujours fixé à 20km


def compute_predicted_end(
    df: pd.DataFrame,
    threshold: float,
    proba_col: str = "proba_lissee",
    confirmation_window: int = 10,
) -> pd.Series:
    """
    t̂^a : première minute confirmée où l'IA déclare la fin de l'orage.
    Score de confiance s^a_i = 1 - proba_lissee.
    On ouvre quand les `confirmation_window` dernières minutes sont toutes < threshold.
    L'IA ne peut ouvrir qu'après le déclenchement de l'alerte humaine (20km).
    Retourne une Série indexée par storm_group_id → position entière. NaN si jamais ouvert.
    """
    grp = df["storm_group_id"]
    alerte_active = (df[ALERT_COL] > 0).astype(int).groupby(grp).cummax().astype(bool)

    danger_brut = (df[proba_col] >= threshold).astype(int)
    danger_confirme = danger_brut.groupby(grp).transform(
        lambda x: x.rolling(window=confirmation_window, min_periods=1).max()
    )

    ouverture = alerte_active & (danger_confirme == 0)

    df_tmp = pd.DataFrame({
        "storm_group_id": df["storm_group_id"].values,
        "ouverture": ouverture.values,
        "row_pos": np.arange(len(df)),
    })
    return df_tmp[df_tmp["ouverture"]].groupby("storm_group_id")["row_pos"].first()


def compute_last_strike_pos(df: pd.DataFrame) -> pd.Series:
    """
    t^a : position du dernier impact CG à 20km par alerte.
    """
    df_tmp = pd.DataFrame({
        "storm_group_id": df["storm_group_id"].values,
        "has_strike": (df[ALERT_COL].values > 0),
        "row_pos": np.arange(len(df)),
    })
    return df_tmp[df_tmp["has_strike"]].groupby("storm_group_id")["row_pos"].last()


def compute_gain_per_alert(
    df: pd.DataFrame,
    threshold: float,
    proba_col: str = "proba_lissee",
    confirmation_window: int = 10,
    horizon_min: int = 30,
) -> pd.Series:
    """
    g^a = t^a + 30 - t̂^a  (en minutes, 1 ligne = 1 minute)
    Si jamais ouvert → g^a = 0. Si ouverture trop tardive → clippé à 0.
    """
    t_last     = compute_last_strike_pos(df)
    t_hat      = compute_predicted_end(df, threshold, proba_col, confirmation_window)
    t_baseline = t_last + horizon_min

    t_hat = t_hat.reindex(t_baseline.index).fillna(t_baseline)  # pas de prédiction → baseline
    return (t_baseline - t_hat).clip(lower=0)


def compute_risk(
    df: pd.DataFrame,
    threshold: float,
    audit_zone_km: int,
    proba_col: str = "proba_lissee",
    confirmation_window: int = 10,
) -> float:
    """
    R = M^{Lk} / N^{Lk}
    N^{Lk} = total impacts CG à < k km dans le dataset.
    M^{Lk} = impacts CG à < k km survenus APRÈS t̂^a dans la même alerte.
    """
    audit_col = f"count_cg_{audit_zone_km}km"
    N_Lk = df[audit_col].sum()
    if N_Lk == 0:
        return 0.0

    t_hat = compute_predicted_end(df, threshold, proba_col, confirmation_window)

    df_tmp = pd.DataFrame({
        "storm_group_id": df["storm_group_id"].values,
        "audit_count": df[audit_col].values,
        "row_pos": np.arange(len(df)),
    }).join(t_hat.rename("t_hat"), on="storm_group_id")

    df_tmp["t_hat"] = df_tmp["t_hat"].fillna(df_tmp["row_pos"] + 1)  # jamais ouvert → rien raté
    missed = df_tmp.loc[df_tmp["row_pos"] > df_tmp["t_hat"], "audit_count"].sum()

    return float(missed / N_Lk)

def find_best_threshold(
    df: pd.DataFrame,
    safety_zones_km: list[int],
    proba_col: str = "proba_lissee",
    risk_threshold: float = 0.02,
    confirmation_window: int = 10,
    horizon_min: int = 30,
    n_thresholds: int = 300,
) -> dict:
    """
    Pour chaque zone d'audit : cherche θ qui maximise G sous contrainte R ≤ risk_threshold.
    Alerte et gain → toujours 20km. Risque → zone demandée.
    """
    results = {}

    for zone_km in safety_zones_km:
        best_threshold, best_gain, best_risk, best_gains = 0.0, -1, 1.0, None

        for theta in np.linspace(0.01, 0.99, n_thresholds):
            R = compute_risk(df, theta, audit_zone_km=zone_km,
                             proba_col=proba_col, confirmation_window=confirmation_window)
            if R <= risk_threshold:
                gains = compute_gain_per_alert(df, theta, proba_col, confirmation_window, horizon_min)
                G = int(gains.sum())
                if G > best_gain:
                    best_gain, best_threshold, best_risk, best_gains = G, theta, R, gains

        results[zone_km] = {
            "threshold": best_threshold,
            "total_gain": best_gain,
            "risk": best_risk,
            "stats": {
                "mean": best_gains.mean(), "median": best_gains.median(),
                "std": best_gains.std(), "max": best_gains.max(), "min": best_gains.min(),
                "q25": best_gains.quantile(0.25), "q75": best_gains.quantile(0.75),
            } if best_gains is not None else None,
            "gains_per_alert": best_gains,
        }

    return results


def print_evaluation_report(
    results: dict,
    zone_km: int,
) -> None:
    """Affiche un rapport d’évaluation formaté."""
    r = results.get(zone_km)
    if r is None or r["total_gain"] < 0:
        print(f"Zone {zone_km}km : aucun seuil valide trouvé.")
        return

    print("=" * 70)
    print(f"Rapport d’évaluation — zone d’audit : {zone_km} km")
    print("=" * 70)
    print(f"  Seuil            : {r['threshold']*100:.3f}%")
    print(f"  Gain total       : {int(r['total_gain'])} minutes")
    print(f"  Risque R         : {r['risk']*100:.3f}%")
    print("-" * 70)
    if r["stats"]:
        s = r["stats"]
        print(f"  Gain moyen/orage : {s['mean']:.1f} min")
        print(f"  Gain médian      : {s['median']:.1f} min")
        print(f"  Écart-type       : {s['std']:.1f} min")
        print(f"  Max / Min        : {s['max']:.1f} / {s['min']:.1f} min")
    print("=" * 70)