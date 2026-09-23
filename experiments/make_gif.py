"""Probe GIF: cumulative accuracy and cost race over 100 cases.
Saves figs/race.gif."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

df = pd.read_csv("figs/probe.csv")
STRATS = ["probe", "cheapest2", "random2"]
COLS = {"probe": "#27ae60", "cheapest2": "#e74c3c", "random2": "#95a5a6"}
cum = {s: df[df.strategy == s].reset_index(drop=True) for s in STRATS}
N = min(len(cum[s]) for s in STRATS)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
for ax in (a1, a2):
    ax.set_xlim(0, N)
a1.set_ylim(0, 1.02)
a1.set_ylabel("cumulative accuracy")
a1.set_xlabel("cases")
a2.set_ylabel("cumulative cost")
a2.set_xlabel("cases")
a2.set_ylim(0, float(max(cum[s].cost.cumsum().iloc[-1] for s in STRATS)) * 1.05)
lines1 = {s: a1.plot([], [], color=COLS[s], label=s)[0] for s in STRATS}
lines2 = {s: a2.plot([], [], color=COLS[s], label=s)[0] for s in STRATS}
a1.legend(fontsize=8)
a2.legend(fontsize=8)
STEPS = 20


def draw(f):
    n = (f + 1) * N // STEPS
    for s in STRATS:
        d = cum[s].iloc[:n]
        lines1[s].set_data(range(n), d.correct.cumsum() / np.arange(1, n + 1))
        lines2[s].set_data(range(n), d.cost.cumsum())
    a1.set_title(f"accuracy after {n} cases")
    return list(lines1.values()) + list(lines2.values())


FuncAnimation(fig, draw, frames=STEPS, interval=400).save(
    "figs/race.gif", writer="pillow", dpi=100)
print("saved figs/race.gif")
