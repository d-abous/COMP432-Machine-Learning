import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # or write your own split
from .config import TRAIN_PATH, TEST_PATH, RANDOM_SEED, VAL_SPLIT

def load_data():
    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y

def load_test():
    df = pd.read_csv(TEST_PATH)
    X_test = df.values  # assuming only features
    ids = np.arange(len(X_test))
    return X_test, ids

def train_val_split(X, y):
    return train_test_split(
        X, y,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y
    )
