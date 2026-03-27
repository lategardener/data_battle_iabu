"""
Identification des groupes de tempêtes.
"""

import pandas as pd

def add_storm_groups_by_target(
    df: pd.DataFrame,
    context_size: int = 20,
    time_window_minutes: int = 60,
) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df.sort_values(["airport", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["storm_group_id"] = -1

    is_target = df["is_last_lightning_cloud_ground"].notna()
    if not is_target.any():
        return

    is_true = df["is_last_lightning_cloud_ground"].isin([True, 1, 1.0, "True"])
    airport_changed = df["airport"] != df["airport"].shift(1)
    after_true = is_true.shift(1).fillna(False)
    raw_group_id = (airport_changed | after_true).cumsum()

    valid_indices: list[int] = []

    for _, group_df in df.groupby(raw_group_id):
        targets = group_df["is_last_lightning_cloud_ground"].notna()
        if not targets.any():
            continue

        first_target_idx = targets.idxmax()
        first_target_time = group_df.loc[first_target_idx, "date"]

        if first_target_idx == group_df.index[0]:
            before_df = pd.DataFrame(columns=group_df.columns)
        else:
            before_df = group_df.loc[: first_target_idx - 1]

        context_indices: list[int] = []
        if not before_df.empty:
            time_diffs = first_target_time - before_df["date"]
            valid_context = before_df[
                time_diffs <= pd.Timedelta(minutes=time_window_minutes)
            ]
            context_indices = valid_context.index[-context_size:].tolist()

        alert_indices = group_df.loc[first_target_idx:].index.tolist()
        valid_indices.extend(context_indices + alert_indices)

    df.loc[valid_indices, "storm_group_id"] = raw_group_id[valid_indices]

    valid_mask = df["storm_group_id"] != -1
    if valid_mask.any():
        df.loc[valid_mask, "storm_group_id"] = (
            pd.factorize(df.loc[valid_mask, "storm_group_id"])[0] + 1
        )

    print(f"Storm groups assigned: {df['storm_group_id'].nunique() - 1} groups found.")