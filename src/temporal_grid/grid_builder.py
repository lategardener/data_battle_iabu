"""
Constructeur de grille temporelle.
Génère UNIQUEMENT les 29 features finales et les cibles pour chaque zone de sécurité.
"""

import numpy as np
import pandas as pd


def build_temporal_grid(
        df_raw: pd.DataFrame,
        horizon_min: int = 30,
        safety_zones_km=None,
) -> pd.DataFrame:
    if safety_zones_km is None:
        safety_zones_km = [20]
    print(f"Construction de la grille temporelle (horizon={horizon_min}m, zones={safety_zones_km}km)...")

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_ic"] = df["icloud"].astype(int)
    df["is_cg"] = (~df["icloud"].astype(bool)).astype(int)
    df["is_positive"] = (df["amplitude"] > 0).astype(int)

    all_grids: list[pd.DataFrame] = []

    # Colonnes finales à conserver
    feature_columns = [
        "minutes_since_last_strike", "cum_n_strikes", "activity_count_last_5m",
        "dist_last_5m", "delta_dist_last_5m", "abs_amplitude_last_5m",
        "activity_count_last_20m", "dist_last_20m", "distance_macro_trend",
        "activity_drop_ratio", "vitesse_eloignement", "dist_projetee_30m",
        "dist_centroid_actuel", "dist_centroid_proj_30m", "centroid_eloignement_net",
        "count_ic_last_20m", "count_cg_last_20m", "count_pos_last_20m",
        "ratio_ic_cg_last_20m", "ratio_pos_last_20m", "hour_sin", "hour_cos",
        "month_sin", "month_cos", "azimuth_std_last_5m", "storm_spread_radial",
        "energie_last_5m", "energy_drop_ratio", "last_known_dist"
    ]

    for (airport_id, storm_id), df_storm in df.groupby(["airport_id", "storm_group_id"]):
        df_storm = df_storm.sort_values("date").set_index("date")

        # Masques pour chaque zone de sécurité
        for zone in safety_zones_km:
            df_storm[f"is_cg_{zone}km"] = ((df_storm["is_cg"] == 1) & (df_storm["dist"] <= zone)).astype(int)

        # Padding basé sur la plus grande zone
        max_zone = max(safety_zones_km)
        mask_danger = df_storm[f"is_cg_{max_zone}km"] == 1
        if mask_danger.any():
            last_danger = df_storm[mask_danger].index.max()
            grid_end = last_danger + pd.Timedelta(minutes=horizon_min)
        else:
            grid_end = df_storm.index.max()

        agg_map = {
            "dist": "min", "delta_dist": "mean", "abs_amplitude": "max",
            "cum_n_strikes": "max", "X": "mean", "Y": "mean",
        }
        df_min = df_storm.resample("1min").agg(agg_map)
        df_min["activity_count"] = df_storm.resample("1min").size()

        new_index = pd.date_range(start=df_min.index.min(), end=grid_end, freq="1min")
        df_min = df_min.reindex(new_index)

        df_min["a_frappe"] = df_min["activity_count"] > 0
        df_min["heure_dernier"] = df_min.index.to_series().where(df_min["a_frappe"]).ffill()
        df_min["minutes_since_last_strike"] = (
                    (df_min.index - df_min["heure_dernier"]).dt.total_seconds() / 60.0).fillna(0)

        df_min["count_ic"] = df_storm["is_ic"].resample("1min").sum().reindex(new_index, fill_value=0)
        df_min["count_cg"] = df_storm["is_cg"].resample("1min").sum().reindex(new_index, fill_value=0)
        df_min["count_pos"] = df_storm["is_positive"].resample("1min").sum().reindex(new_index, fill_value=0)
        df_min["dist_max"] = df_storm["dist"].resample("1min").max().reindex(new_index)
        df_min["azimuth_std"] = df_storm["azimuth"].resample("1min").std().fillna(0).reindex(new_index, fill_value=0)
        df_min["energie_totale"] = df_storm["abs_amplitude"].resample("1min").sum().reindex(new_index, fill_value=0)

        df_min["last_known_dist"] = df_min["dist"].ffill()
        df_min["cum_n_strikes"] = df_min["cum_n_strikes"].ffill().fillna(0)
        df_min[["abs_amplitude", "activity_count"]] = df_min[["abs_amplitude", "activity_count"]].fillna(0)

        feat = df_min[["minutes_since_last_strike", "cum_n_strikes", "last_known_dist"]].copy()

        df_5m = df_min.rolling(window=5, min_periods=1).agg({
            "activity_count": "sum", "dist": "min", "delta_dist": "mean",
            "abs_amplitude": "max", "X": "mean", "Y": "mean",
        })
        df_5m.columns = [f"{c}_last_5m" for c in df_5m.columns]

        df_20m = df_min.rolling(window=20, min_periods=1).agg({
            "activity_count": "sum", "dist": "min",
        })
        df_20m.columns = [f"{c}_last_20m" for c in df_20m.columns]

        feat = pd.concat([feat, df_5m, df_20m], axis=1)

        feat["hour_sin"] = np.sin(2 * np.pi * feat.index.hour / 24.0)
        feat["hour_cos"] = np.cos(2 * np.pi * feat.index.hour / 24.0)
        feat["month_sin"] = np.sin(2 * np.pi * feat.index.month / 12.0)
        feat["month_cos"] = np.cos(2 * np.pi * feat.index.month / 12.0)

        feat["dist_last_5m"] = feat["dist_last_5m"].fillna(df_min["last_known_dist"])
        feat["dist_last_20m"] = feat["dist_last_20m"].fillna(df_min["last_known_dist"])

        feat["distance_macro_trend"] = feat["dist_last_5m"] - feat["dist_last_20m"]
        feat["activity_drop_ratio"] = feat["activity_count_last_5m"] / (feat["activity_count_last_20m"] + 0.001)
        feat["vitesse_eloignement"] = feat["distance_macro_trend"] / 15.0
        feat["dist_projetee_30m"] = feat["dist_last_5m"] + feat["vitesse_eloignement"] * 30.0

        vitesse_X = ((feat["X_last_5m"] - feat["X_last_5m"].shift(5)) / 5.0).fillna(0)
        vitesse_Y = ((feat["Y_last_5m"] - feat["Y_last_5m"].shift(5)) / 5.0).fillna(0)

        feat["dist_centroid_actuel"] = np.sqrt(feat["X_last_5m"] ** 2 + feat["Y_last_5m"] ** 2)
        proj_X_30m = feat["X_last_5m"] + vitesse_X * 30.0
        proj_Y_30m = feat["Y_last_5m"] + vitesse_Y * 30.0
        feat["dist_centroid_proj_30m"] = np.sqrt(proj_X_30m ** 2 + proj_Y_30m ** 2)
        feat["centroid_eloignement_net"] = feat["dist_centroid_proj_30m"] - feat["dist_centroid_actuel"]

        feat["count_ic_last_20m"] = df_min["count_ic"].rolling(window=20, min_periods=1).sum()
        feat["count_cg_last_20m"] = df_min["count_cg"].rolling(window=20, min_periods=1).sum()
        feat["count_pos_last_20m"] = df_min["count_pos"].rolling(window=20, min_periods=1).sum()
        feat["ratio_ic_cg_last_20m"] = feat["count_ic_last_20m"] / (feat["count_cg_last_20m"] + 0.001)
        feat["ratio_pos_last_20m"] = feat["count_pos_last_20m"] / (feat["activity_count_last_20m"] + 0.001)

        dist_max_last_5m = df_min["dist_max"].rolling(window=5, min_periods=1).max()
        feat["azimuth_std_last_5m"] = df_min["azimuth_std"].rolling(window=5, min_periods=1).mean()
        feat["storm_spread_radial"] = dist_max_last_5m - feat["dist_last_5m"]

        feat["energie_last_5m"] = df_min["energie_totale"].rolling(window=5, min_periods=1).sum()
        energie_last_20m = df_min["energie_totale"].rolling(window=20, min_periods=1).sum()
        feat["energy_drop_ratio"] = feat["energie_last_5m"] / (energie_last_20m + 0.001)

        zero_fill = ["distance_macro_trend", "vitesse_eloignement", "centroid_eloignement_net", "storm_spread_radial"]
        feat[zero_fill] = feat[zero_fill].fillna(0)

        dist_fallback = df_min["last_known_dist"]
        for col in ["dist_projetee_30m", "dist_centroid_actuel", "dist_centroid_proj_30m"]:
            feat[col] = feat[col].fillna(dist_fallback)
        feat["last_known_dist"] = dist_fallback

        # Cibles + Comptages pour l'évaluation par zones
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_min)
        for zone in safety_zones_km:
            # Comptage du nombre de CG par minute dans la zone
            count_col = df_storm[f"is_cg_{zone}km"].resample("1min").sum().reindex(feat.index, fill_value=0)
            feat[f"count_cg_{zone}km"] = count_col
            # Si un CG apparait dans l'horizon, target = 1
            future_danger = count_col.rolling(window=indexer, min_periods=1).sum()
            feat[f"target_{horizon_min}m_{zone}km"] = (future_danger > 0).astype(int)

        feat["airport_id"] = airport_id
        feat["storm_group_id"] = storm_id
        feat = feat.reset_index().rename(columns={"index": "date"})

        # Filtrage final : On garde Date, IDs, les 29 features exactes, les cibles et les compteurs d'audit
        keep_cols = ["date", "airport_id", "storm_group_id"] + feature_columns + \
                    [f"target_{horizon_min}m_{z}km" for z in safety_zones_km] + \
                    [f"count_cg_{z}km" for z in safety_zones_km]

        feat = feat[[c for c in keep_cols if c in feat.columns]]
        all_grids.append(feat)

    print("Assemblage de la grille finale...")
    result = pd.concat(all_grids, ignore_index=True)
    return result