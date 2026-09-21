"""Probe demo: case player. Fit once at startup, then interactive.
Run: <venv-python> demo/app.py (port 5004)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/dead/pfn-probe/src")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from probe.engine import Prober
from src.ev import load, stratified_subsample

import sys as _s
_s.path.insert(0, "/home/dead/pfn-probe/experiments")
from run import CATS, COSTS, FREE, decide

print("fitting prober...", flush=True)
X, y, _, _, _ = load("/home/dead/playground-series-s6e9/data")
Xs, ys = stratified_subsample(X, y, 1500, seed=0)
Xctx, Xte, yctx, yte = train_test_split(Xs, ys, test_size=30, stratify=ys,
                                        random_state=1)
PB = Prober(Xctx.reset_index(drop=True), yctx, COSTS, cat_cols=CATS, lam=0.05,
            seed=0)
CASES = [(Xte.iloc[i].to_dict(), int(yte[i])) for i in range(len(Xte))]
STATE = {"i": 0, "known": list(FREE), "cost": 0.0}
print("ready.", flush=True)

app = Flask(__name__)
PAGE = """
<h1>PFN Probe — know what to measure, know when to stop</h1>
<form method=get>Case: <select name=i>{% for k in range(n) %}<option {{'selected' if k==i}}>{{k}}</option>{% endfor %}</select>
<input type=submit name=a value="load">
<input type=submit name=a value="reset"></form>
<h2>Known</h2><ul>{% for k, v in known %}<li>{{k}}: {{v}}</li>{% endfor %}</ul>
<h2>Risk {{'%.0f' % (100*p)}}% — {{dec}}</h2>
<h3>What next?</h3>
<table border=1 cellpadding=4><tr><th>test</th><th>E[gain]</th><th>cost</th><th>net</th><th></th></tr>
{% for j, g, c, v in cards %}<tr><td>{{j}}</td><td>{{'%.3f' % g}}</td><td>{{c}}</td>
<td><b>{{'%.3f' % v}}</b></td>
<td><a href="/?i={{i}}&a=order&j={{j}}">order</a></td></tr>{% endfor %}</table>
<p>{{msg}}</p>
"""


def blank(full, known):
    r = {k: (v if k in known else np.nan) for k, v in full.items()}
    return pd.DataFrame([r])


@app.get("/")
def index():
    i = int(request.args.get("i", STATE["i"]))
    a = request.args.get("a", "")
    if a == "reset" or i != STATE["i"]:
        STATE.update(i=i, known=list(FREE), cost=0.0)
    full, truth = CASES[i]
    if a == "order":
        j = request.args.get("j")
        STATE["known"].append(j)
        STATE["cost"] += COSTS[j]
    known = STATE["known"]
    row = blank(full, known)
    j, p, vois = PB.step(row, known)
    cards = [(k, vois[k] + COSTS[k], COSTS[k], vois[k])
             for k in sorted(vois, key=vois.get, reverse=True)]
    if j is None:
        dec = "STOP — sufficient evidence" if p < 0.5 else "STOP — decide POSITIVE"
        msg = f"total measurement cost {STATE['cost']}"
    else:
        dec = "INSUFFICIENT INFORMATION"
        msg = f"recommended: order {j} (net {vois[j]:.3f}); spent {STATE['cost']}"
    kv = [(k, full[k] if k in known else "?") for k in full]
    return render_template_string(PAGE, n=len(CASES), i=i, known=kv, p=p, dec=dec,
                                  cards=cards, msg=msg)


if __name__ == "__main__":
    app.run(debug=True, port=5004)
