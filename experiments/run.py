"""Probe trajectories vs baselines (S6E9-sub). Costs illustrative.
Saves figs/probe.csv (per-row: strategy, correct, cost, n_tests). GPU ~40 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/dead/pfn-probe/src")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from probe.engine import Prober, bayes_risk
from src.ev import load, stratified_subsample

SEED = 0
COSTS = {"Age": 0, "Gender": 0, "City_Type": 0, "Daily_Commute_km": 1,
         "Number_of_Cars_Owned": 1, "Environmental_Concern_Level": 1,
         "Charging_Stations_Near_Home": 2, "Charging_Stations_Near_Work": 2,
         "Current_Car_Type": 2, "Range_Anxiety_Level": 2,
         "Home_Charging_Possible": 3, "Subsidy_Available": 3,
         "Annual_Income_USD": 5}
FREE = ["Age", "Gender", "City_Type"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]


def decide(p, c_fn=5.0, c_fp=1.0):
    return int(p * c_fn > (1 - p) * c_fp)


def run_row(pb: Prober, full: pd.Series, truth: int, mode: str):
    row = full.copy()
    known = list(FREE)
    for c in row.index:
        if c not in known:
            row[c] = np.nan
    row = pd.DataFrame([row])
    cost, n = 0.0, 0
    order = None
    if mode == "cheapest":
        order = sorted([c for c in row.columns if c not in known],
                       key=lambda c: COSTS[c])
    elif mode == "random":
        order = list(np.random.RandomState(9).permutation(
            [c for c in row.columns if c not in known]))
    while True:
        if mode == "probe":
            j, p, _ = pb.step(row, known)
            if j is None:
                break
        elif mode == "all":
            j = next((c for c in row.columns if c not in known), None)
            if j is None:
                break
            p = pb._predict1(row)
        else:
            j = order[n] if n < len(order) else None
            if j is None:
                break
            p = pb._predict1(row)
        row[j] = full[j]
        known.append(j)
        cost += COSTS[j]
        n += 1
    p = pb._predict1(row)
    d = decide(p)
    loss = (d != truth) * (5.0 if truth == 1 else 1.0)
    return d == truth, cost, n, loss + cost


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y, _, _, _ = load("/home/dead/playground-series-s6e9/data")
    Xs, ys = stratified_subsample(X, y, 2000, seed=SEED)
    Xctx, Xte, yctx, yte = train_test_split(Xs, ys, test_size=20,
                                            stratify=ys, random_state=1)
    pb = Prober(Xctx.reset_index(drop=True), yctx, COSTS, cat_cols=CATS,
                lam=0.05, seed=SEED)
    recs = []
    cols = ["row", "strategy", "correct", "cost", "n_tests", "total"]
    for i in range(len(Xte)):
        full = Xte.iloc[i]
        for mode in ["probe", "all", "cheapest", "random"]:
            ok, cost, n, tot = run_row(pb, full, int(yte[i]), mode)
            recs.append((i, mode, ok, cost, n, tot))
            print(f"row {i} {mode}: ok={ok} cost={cost} n={n}", flush=True)
        pd.DataFrame(recs, columns=cols).to_csv("figs/probe.csv", index=False)
    df = pd.DataFrame(recs, columns=cols)
    df.to_csv("figs/probe.csv", index=False)
    g = df.groupby("strategy").agg(acc=("correct", "mean"), cost=("cost", "mean"),
                                   n=("n_tests", "mean"), total=("total", "mean"))
    print(g.round(3).to_string(), flush=True)
    print(f"saved figs/probe.csv ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
