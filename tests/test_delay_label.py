# tests/test_delay_label.py

import pandas as pd


def add_delay_label(df: pd.DataFrame, threshold: int = 15) -> pd.DataFrame:
    """
    Minimal version of the label logic used in the project:
    Delayed = 1 if arrival delay >= threshold, else 0.
    """
    df = df.copy()
    df["Delayed"] = (df["ArrDelay"] >= threshold).astype(int)
    return df


def test_delay_label_logic():
    """Check that the Delayed label matches the >= 15 minutes rule."""
    data = {"ArrDelay": [-5, 0, 10, 15, 20, 30]}
    df = pd.DataFrame(data)

    labeled = add_delay_label(df, threshold=15)

    # Expected: only delays >= 15 are marked as 1
    expected = [0, 0, 0, 1, 1, 1]
    assert labeled["Delayed"].tolist() == expected
