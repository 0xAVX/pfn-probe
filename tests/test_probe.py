import numpy as np
import pandas as pd

from probe.engine import Prober, bayes_risk


def toy():
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n),
                       "c": rng.choice(["x", "y"], n)})
    return df, ((df.a + (df.c == "x")) > 0).astype(int).values


def test_voi_and_stop():
    X, y = toy()
    pb = Prober(X.iloc[:300], y[:300], {"a": 0, "b": 1, "c": 5},
                cat_cols=["c"], seed=0)
    row = pd.DataFrame([{"a": 0.5, "b": np.nan, "c": np.nan}])
    p0, r0, vois = pb.voi(row, ["a"], ["b", "c"])
    assert set(vois) == {"b", "c"} and np.isfinite(list(vois.values())).all()
    j, p, _ = pb.step(row, ["a", "b", "c"])
    assert j is None  # nothing left -> STOP
    import math
    assert math.isclose(bayes_risk(0.9), 0.1, rel_tol=1e-9)
