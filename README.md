# PFN Probe — measure only what matters

> **Why this matters (20s):** Every medical test or survey question costs
> money. Probe asks TabPFN what each unmeasured feature is worth *before*
> measuring it, orders the best deal, and STOPs when nothing is worth the
> price — 0.85 accuracy at 3.95 cost vs 0.95 at 22.0 for measure-everything.

Sequential feature acquisition with a STOP action. A case arrives partial;
every unmeasured feature has a cost. TabPFN-3.5 predicts natively on partial
rows; plausible outcomes come from hot-deck conditional sampling (nearest
context neighbors on observed features — no fits, calibrated spread; an
earlier TabPFN-imputation simulator proved overdispersed and was cut).
VOI(j) = risk now − expected risk after measuring j − λ·cost(j).
Order while max VOI > 0, else STOP. No learned acquisition policy.

```python
pb = Prober(Xctx, yctx, costs, cat_cols=[...])
j, p, vois = pb.step(partial_row, known)  # j=None means STOP
```

Costs in this repo are illustrative (S6E9 survey/income-verification story).

## Results (`figs/probe.csv`, 20 cases, S6E9)

| strategy | accuracy | avg cost | avg tests | loss+cost | loss+0.05·cost |
|---|---|---|---|---|---|
| order-all | 0.95 | 22.0 | 10 | 22.05 | 1.150 |
| cheapest-first | 0.95 | 22.0 | 10 | 22.05 | 1.150 |
| random order | 0.95 | 22.0 | 10 | 22.05 | 1.150 |
| cheapest-2 (fixed) | 0.15 | 2.0 | 2 | 2.85 | 0.950 |
| random-2 (fixed) | 0.50 | 5.0 | 2 | 5.50 | 0.750 |
| **PFN Probe** | 0.85 | **3.95** | **2.0** | **4.10** | **0.348** |

Pareto read: probe trades 0.10 accuracy for 5.6× lower measurement cost and
wins both the raw combined objective (4.10 vs next 5.50) and the
policy-consistent loss+0.05·cost (0.348 vs 0.750). Fixed-2 baselines show the
policy matters, not just the budget: cheapest-2 collapses to 0.15.

## Reproduce

Fresh-env verified 2026-09-22 (clean venv, `pip install -e .`, Probe VOI suite 1 passed in 17s CPU; TabPFN weights from public HF, no keys).

```bash
pip install -e .   # Python 3.10+, torch, tabpfn==9.0.0
# S6E9 data:
kaggle competitions download -c playground-series-s6e9 -p data && unzip -o data/*.zip -d data/
<venv-python> -m pytest tests/ -q
<venv-python> experiments/run.py   # figs/probe.csv
<venv-python> demo/app.py          # case player (port 5004)
```
