# src/utils/data_loader.py
import pandas as pd
import os
from src.config import PITCH_LENGTH, PITCH_WIDTH


# def load_match_data(match_name):
def load_match_data(path):
    # path = f"data/{match_name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    df = pd.read_csv(path)
    df = preprocess_data(df)
    return df


def preprocess_data(df):
    df['x'] = df['x'] / 100 * PITCH_LENGTH
    df['y'] = df['y'] / 100 * PITCH_WIDTH
    df['end_x'] = df['end_x'] / 100 * PITCH_LENGTH
    df['end_y'] = df['end_y'] / 100 * PITCH_WIDTH
    return df
