"""
Fonctions pour générer des rapports détaillés et visualiser les courbes de probabilité.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List


def print_detailed_report(
    results: dict,
    zone_km: int,
    training_threshold: Optional[float] = None,
    airports: Optional[List[str]] = None,
) -> None:
    """
    Affiche un rapport détaillé pour une zone avec statistiques complètes.

    Arguments:
        results : Résultats du find_best_threshold ou predict_batch
        zone_km : Zone d'audit à rapporter (3, 5, 7, 10, 15, ou 20)
        training_threshold : Seuil utilisé lors de l'entraînement (pour contexte)
        airports : Liste des aéroports à inclure (None = tous les aéroports)
    """
    r = results.get(zone_km)
    if r is None or r.get("total_gain", -1) < 0:
        print(f"❌ Zone {zone_km}km : aucun seuil valide trouvé.")
        return

    print("\n" + "=" * 80)
    print(f"  📊 RAPPORT DÉTAILLÉ — Zone d'audit : {zone_km} km")
    print("=" * 80)

    # Section 1 : Configuration
    print("\n  ⚙️  CONFIGURATION")
    print("  " + "-" * 76)
    if training_threshold is not None:
        print(f"  • Seuil d'entraînement        : {training_threshold*100:>7.3f}%")
    print(f"  • Seuil de test               : {r['threshold']*100:>7.3f}%")
    if airports:
        print(f"  • Aéroports filtrés            : {', '.join(airports)}")
    else:
        print(f"  • Aéroports                    : Tous")

    # Section 2 : Résultats globaux
    print("\n  📈 RÉSULTATS GLOBAUX")
    print("  " + "-" * 76)
    print(f"  • Gain total                  : {int(r['total_gain']):>7} minutes")
    print(f"  • Risque R                    : {r['risk']*100:>7.3f}%")

    # Section 3 : Statistiques par orage
    if r.get("stats"):
        s = r["stats"]
        print("\n  📉 STATISTIQUES PAR ORAGE")
        print("  " + "-" * 76)
        print(f"  • Gain moyen                  : {s['mean']:>7.1f} minutes")
        print(f"  • Gain médian                 : {s['median']:>7.1f} minutes")
        print(f"  • Écart-type                  : {s['std']:>7.1f} minutes")
        print(f"  • Gain minimum                : {s['min']:>7.1f} minutes")
        print(f"  • Gain maximum                : {s['max']:>7.1f} minutes")
        print(f"  • Q25 (1er quartile)          : {s['q25']:>7.1f} minutes")
        print(f"  • Q75 (3e quartile)           : {s['q75']:>7.1f} minutes")

    print("\n" + "=" * 80 + "\n")


def get_stats_filtered_by_airports(
    df: pd.DataFrame,
    threshold: float,
    zone_km: int,
    airports: Optional[List[str]] = None,
    proba_col: str = "proba_lissee",
    confirmation_window: int = 10,
) -> dict:
    """
    Calcule les statistiques pour une zone et un seuil donnés,
    en filtrant optionnellement par aéroport.

    Arguments:
        df : DataFrame avec les prédictions et les données
        threshold : Seuil de probabilité à tester
        zone_km : Zone d'audit (3, 5, 7, 10, 15, 20)
        airports : Liste des aéroports (None = tous)
        proba_col : Colonne contenant les probabilités lissées
        confirmation_window : Fenêtre de confirmation pour les décisions

    Retourne:
        dict avec clés 'mean', 'median', 'std', 'min', 'max', 'q25', 'q75'
    """
    from src.evaluation.metrics import compute_gain_per_alert

    df_filtered = df.copy()
    if airports:
        df_filtered = df_filtered[df_filtered['airport'].isin(airports)]

    if df_filtered.empty:
        print(f"⚠️  Aucune donnée trouvée pour les aéroports : {airports}")
        return {}

    gains = compute_gain_per_alert(
        df_filtered,
        threshold=threshold,
        proba_col=proba_col,
        confirmation_window=confirmation_window,
    )

    if gains.empty:
        return {}

    return {
        'mean': gains.mean(),
        'median': gains.median(),
        'std': gains.std(),
        'min': gains.min(),
        'max': gains.max(),
        'q25': gains.quantile(0.25),
        'q75': gains.quantile(0.75),
    }


def plot_probability_curve(
    df: pd.DataFrame,
    storm_id: int,
    threshold: float,
    safety_zone_km: int = 20,
    confirmation_window: int = 10,
    horizon_min: int = 30,
) -> None:
    """
    Trace une courbe de probabilité pour un orage spécifique avec :
    - Zone VERTE : où l'alerte aurait pu être levée (proba < seuil)
    - Zone BLEUE : où l'alerte a été effectivement levée
    - Bâtonnets ROUGES : CG au sol dans la zone
    - Bâtonnets JAUNES : IC (nuages) dans la zone
    - Bâtonnets VIOLETS : Éclairs en dehors de la zone

    Arguments:
        df : DataFrame avec prédictions et données brutes
        storm_id : ID du groupe d'orage à visualiser
        threshold : Seuil de décision
        safety_zone_km : Zone de sécurité à afficher (3, 5, 7, 10, 15, 20)
        confirmation_window : Fenêtre de confirmation pour les décisions
        horizon_min : Horizon de prédiction en minutes
    """
    from src.evaluation.metrics import ALERT_COL

    df_s = df[df['storm_group_id'] == storm_id].copy().reset_index(drop=True)

    if df_s.empty:
        print(f"❌ Orage #{storm_id} introuvable.")
        return

    # Filtrer pour commencer après le premier impact CG
    if df_s[ALERT_COL].sum() > 0:
        first_cg = (df_s[ALERT_COL] > 0).idxmax()
        df_s = df_s.loc[first_cg:].reset_index(drop=True)

    t = np.arange(len(df_s))

    # Calcul de la décision d'ouverture
    danger_brut = (df_s[proba_col := 'proba_lissee'] >= threshold).astype(int)
    danger_confirme = danger_brut.rolling(window=confirmation_window, min_periods=1).max()
    ouverture = (danger_confirme == 0).astype(int)

    # Figure
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 1. Courbe principale de probabilité
    ax1.plot(t, df_s[proba_col], color='dodgerblue', linewidth=2.5, label='Probabilité lissée', zorder=3)
    ax1.axhline(threshold, color='darkred', linestyle='--', linewidth=2, label=f'Seuil ({threshold:.3f})', zorder=2)

    # 2. Zone VERTE : où l'alerte aurait pu être levée
    ax1.fill_between(t, 0, 1, where=(ouverture == 1), color='lightgreen', alpha=0.25, label='Zone de levée d\'alerte possible', zorder=1)

    # 3. Zone BLEUE : où l'alerte a réellement été levée
    alert_active = (df_s[ALERT_COL] > 0).cummax().astype(bool)
    alerte_declenchee = alert_active & (ouverture == 0)
    ax1.fill_between(t, 0, 1, where=alerte_declenchee, color='lightskyblue', alpha=0.3, label='Alerte levée', zorder=1)

    # Axe Y (probabilité)
    ax1.set_ylabel('Probabilité de danger', fontsize=11, fontweight='bold', color='dodgerblue')
    ax1.set_ylim(-0.05, 1.10)
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.set_xlabel('Temps depuis le début de l\'orage (minutes)', fontsize=11, fontweight='bold')

    # 4. Créer un axe secondaire pour les éclairs
    ax2 = ax1.twinx()

    # Bâtonnets d'éclairs
    count_col_zone = f"count_cg_{safety_zone_km}km"
    count_col_ic_zone = f"count_ic_{safety_zone_km}km" if f"count_ic_{safety_zone_km}km" in df_s.columns else None

    if count_col_zone in df_s.columns and df_s[count_col_zone].sum() > 0:
        ax2.bar(t, df_s[count_col_zone] * 0.3, color='red', alpha=0.6, width=0.8, label=f'CG au sol < {safety_zone_km}km', zorder=2)

    # Bâtonnets IC (nuages) - jaunes
    if count_col_ic_zone and count_col_ic_zone in df_s.columns and df_s[count_col_ic_zone].sum() > 0:
        ax2.bar(t, df_s[count_col_ic_zone] * 0.2, color='gold', alpha=0.6, width=0.8, label=f'IC (nuages) < {safety_zone_km}km', zorder=2)

    # Éclairs en dehors de la zone (violets) - optionnel
    count_col_outside = f"count_cg_20km"
    if count_col_outside in df_s.columns and count_col_zone in df_s.columns:
        count_outside = df_s[count_col_outside] - df_s[count_col_zone]
        if count_outside.sum() > 0:
            ax2.bar(t, count_outside * 0.15, color='mediumpurple', alpha=0.5, width=0.8, label='Éclairs en dehors', zorder=1)

    ax2.set_ylabel('Nombre d\'éclairs', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, ax2.get_ylim()[1])

    # Annotations : dernière frappe et baseline
    if df_s[ALERT_COL].sum() > 0:
        last_strike_idx = (df_s[ALERT_COL] > 0).idxmax() if ALERT_COL in df_s.columns else df_s[count_col_zone].idxmax()
        last_strike_idx = df_s[df_s[ALERT_COL] > 0].index[-1] if ALERT_COL in df_s.columns else -1

        try:
            last_strike_idx = df_s[df_s[ALERT_COL] > 0].index[-1]
            ax1.axvline(last_strike_idx, color='darkred', linestyle='-.', linewidth=2, alpha=0.7, label='Dernier impact', zorder=2)
            ax1.axvline(last_strike_idx + horizon_min, color='purple', linestyle=':', linewidth=2, alpha=0.7, label='Fin baseline humaine (+ 30 min)', zorder=2)
        except:
            pass

    # Légende et titre
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9, framealpha=0.95)

    ax1.set_title(f'Orage #{storm_id} — Zone d\'audit {safety_zone_km}km — Seuil {threshold:.3f}',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()