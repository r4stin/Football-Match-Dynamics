# src/players_position.py
import pandas as pd


def get_players_position(interval_df):
    """
    Get the average position of each player in the interval.
    :param interval_df:
    :return: player_positions: dict
    """
    player_positions = {}
    for player in set(interval_df["lastName"]).union(interval_df["lastName_next"]):
        player_passes = interval_df[interval_df["lastName"] == player][["x", "y"]]
        player_receptions = interval_df[interval_df["lastName_next"] == player][["end_x", "end_y"]]
        all_positions = pd.concat(
            [player_passes, player_receptions.rename(columns={"end_x": "x", "end_y": "y"})])
        player_positions[player] = all_positions[["x", "y"]].mean().to_numpy()

    return player_positions
