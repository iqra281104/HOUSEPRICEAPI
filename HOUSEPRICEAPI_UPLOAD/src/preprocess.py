"""
preprocess.py

This module is responsible for:
1. Loading the dataset
2. Encoding categorical variables
3. Splitting the data into training and testing sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE


def load_data():
    """
    Load the housing dataset.

    Returns:
        pandas.DataFrame: Housing dataset
    """
    df = pd.read_csv(DATA_PATH)
    return df


def encode_data(df):
    """
    Convert categorical variables into numerical values.

    Parameters:
        df (DataFrame): Housing dataset

    Returns:
        DataFrame: Encoded dataset
    """

    yes_no_columns = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea"
    ]

    for column in yes_no_columns:
        df[column] = df[column].map({
            "yes": 1,
            "no": 0
        })

    df["furnishingstatus"] = df["furnishingstatus"].map({
        "unfurnished": 0,
        "semi-furnished": 1,
        "furnished": 2
    })

    return df


def split_data(df):
    """
    Split the dataset into features and target,
    then into training and testing sets.
    """

    X = df.drop("price", axis=1)
    y = df["price"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )