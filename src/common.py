# src/common.py
import numpy as np
import pandas as pd
from .config import SUBMISSION_DIR

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def save_submission(ids, labels, filename="submission.csv"):
    df = pd.DataFrame({"id": ids, "label": labels.astype(int)})
    df.to_csv(SUBMISSION_DIR + filename, index=False)
