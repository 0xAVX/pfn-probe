"""Self-contained S6E9 loader (no sibling repos, no absolute paths)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATS_S6E9 = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
             "Subsidy_Available", "Range_Anxiety_Level"]


def load_s6e9(data_dir="data", nrows=None):
    data_dir = Path(data_dir)
    tr = pd.read_csv(data_dir / "train.csv", nrows=nrows)
    te = pd.read_csv(data_dir / "test.csv", nrows=nrows)
    feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
    for c in CATS_S6E9:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
    return tr[feats], y, te[feats], te["id"].values, feats


def stratified_subsample(X, y, n, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    n_pos = int(round(n * np.mean(y)))
    take = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(np.where(y == 0)[0], n - n_pos,
                                      replace=False)])
    rng.shuffle(take)
    return X.iloc[take], y[take]
