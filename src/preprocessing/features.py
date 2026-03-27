"""
Feature engineering sur les données brutes.
Ne calcule que les variables intermédiaires requises par la grille temporelle.
"""

import numpy as np
import pandas as pd

STORM_COL = "storm_group_id"

def sort_for_sequences(df: pd.DataFrame) -> None:
    df.sort_values([STORM_COL, "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

def add_cartesian_coordinates(df: pd.DataFrame) -> None:
    df["azimuth_rad"] = np.radians(df["azimuth"])
    df["X"] = df["dist"] * np.sin(df["azimuth_rad"])
    df["Y"] = df["dist"] * np.cos(df["azimuth_rad"])

def add_kinematic_features(df: pd.DataFrame) -> None:
    df["delta_dist"] = df.groupby(STORM_COL)["dist"].diff().fillna(0.0)

def add_intensity_features(df: pd.DataFrame) -> None:
    df["abs_amplitude"] = df["amplitude"].abs()

def add_cumulative_features(df: pd.DataFrame) -> None:
    df["cum_n_strikes"] = df.groupby(STORM_COL).cumcount() + 1