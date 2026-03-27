"""
Nettoyage des données, formatage et suppression du bruit.
"""

import pandas as pd


def format_date(df: pd.DataFrame) -> None:
    """Convertit la colonne 'date' en format datetime."""
    df["date"] = pd.to_datetime(df["date"])


def format_alert_id(df: pd.DataFrame) -> None:
    """Remplit et convertit airport_alert_id en entier."""
    df["airport_alert_id"] = df["airport_alert_id"].fillna(0).astype(int)


def format_last_lightning(df: pd.DataFrame) -> None:
    """
    Convertit is_last_lightning_cloud_ground en entier.
    Doit être appelée APRÈS la création des groupes d’orages.
    """
    df["is_last_lightning_cloud_ground"] = (
        df["is_last_lightning_cloud_ground"].fillna(False).astype(int)
    )


def format_icloud(df: pd.DataFrame) -> None:
    """Convertit le flag icloud en entier (0 = CG, 1 = IC)."""
    df["icloud"] = df["icloud"].astype(int)


def remove_pise_2016(df: pd.DataFrame) -> None:
    """
    Supprime les lignes de l’aéroport de Pise en 2016 (anomalie connue du capteur Meteorage).
    Modifie df en place.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    mask = (df["airport"] == "Pise") & (df["date"].dt.year == 2016)
    n_dropped = mask.sum()

    if n_dropped > 0:
        df.drop(df[mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Nettoyage : {n_dropped} lignes supprimées pour Pise (2016).")
    else:
        print("Aucune anomalie Pise 2016 détectée.")


def remove_noise_data(df: pd.DataFrame) -> None:
    """
    Supprime les lignes n’appartenant à aucun groupe d’orage (storm_group_id == -1).
    Modifie df en place.
    """
    mask = df["storm_group_id"] == -1
    n_dropped = mask.sum()

    if n_dropped > 0:
        df.drop(df[mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Nettoyage : {n_dropped} lignes de bruit supprimées. Restant : {len(df)}")
    else:
        print("Aucune ligne de bruit trouvée.")


def filter_useful_storms(df: pd.DataFrame, alert_zone_km: int = 20) -> None:
    """
    Conserve uniquement les orages avec au moins un impact CG dans la zone d’alerte.
    Les orages sans déclenchement réel d’alerte sont supprimés.

    Args:
        df: DataFrame prétraité avec storm_group_id et icloud.
        alert_zone_km: Rayon définissant la zone d’alerte.
    """
    initial_rows = len(df)
    initial_groups = df["storm_group_id"].nunique()

    is_alert_strike = (df["icloud"] == 0) & (df["dist"] <= alert_zone_km)
    valid_ids = df.loc[is_alert_strike, "storm_group_id"].unique()

    mask = df["storm_group_id"].isin(valid_ids)
    df.drop(df[~mask].index, inplace=True)
    df["storm_group_id"] = pd.factorize(df["storm_group_id"])[0] + 1
    df.reset_index(drop=True, inplace=True)

    print(
        f"Filtrage : {initial_groups} -> {df['storm_group_id'].nunique()} groupes | "
        f"{initial_rows} -> {len(df)} lignes"
    )