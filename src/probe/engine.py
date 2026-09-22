"""PFN Probe engine: sequential feature acquisition with STOP.

TabPFN is the decision model: it predicts natively on partial rows and
evaluates expected decision value. Candidate outcomes come from a
nearest-neighbor hot-deck sampler over context rows (no fits, calibrated
spread) — NOT a TabPFN simulator. VOI(j) = R(x_O) - E[R(x_O,x_j)] - λC_j
with Bayes risk R under asymmetric error costs. STOP when max VOI <= 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier


def bayes_risk(p, c_fn=5.0, c_fp=1.0):
    return min(float(p) * c_fn, (1 - float(p)) * c_fp)


class Prober:
    def __init__(self, Xctx: pd.DataFrame, yctx: np.ndarray, costs: dict,
                 c_fn=5.0, c_fp=1.0, lam=0.1, seed=0, cat_cols=()):
        self.Xctx, self.yctx = Xctx, yctx
        self.costs, self.c_fn, self.c_fp, self.lam = costs, c_fn, c_fp, lam
        self.seed, self.cat = seed, set(cat_cols)
        self.clf = TabPFNClassifier(random_state=seed)
        self.clf.fit(Xctx, yctx)

    def _predict1(self, row: pd.DataFrame) -> float:
        row = row.replace({None: np.nan})
        return float(self.clf.predict_proba(row)[:, 1][0])

    def _outcomes(self, row: pd.DataFrame, j: str, k=5):
        """Hot-deck conditional outcomes: j-values of nearest ctx neighbors
        on observed features. No fits, calibrated spread, honest VOI."""
        obs = [c for c in row.columns if c != j and pd.notna(row[c].iloc[0])]
        if not obs:
            return list(self.Xctx[j].dropna().sample(
                min(k, len(self.Xctx)), random_state=self.seed))
        ref = self.Xctx[obs].copy()
        q = row[obs].iloc[0]
        num = [c for c in obs if c not in self.cat]
        d = np.zeros(len(ref))
        for c in num:
            v = pd.to_numeric(ref[c], errors="coerce").astype(float)
            qv = float(q[c]) if pd.notna(q[c]) else np.nan
            d += ((v - qv) / (v.std() + 1e-9)) ** 2
        for c in obs:
            if c in self.cat or ref[c].dtype == object:
                d += (ref[c].astype(str) != str(q[c])).astype(float) * 2.0
        d = np.where(np.isnan(d), np.nanmax(d) + 1, d)
        nn = np.argsort(d)[:k]
        return self.Xctx[j].iloc[nn].tolist()

    def voi(self, row: pd.DataFrame, known: list, cand: list):
        p0 = self._predict1(row)
        r0 = bayes_risk(p0, self.c_fn, self.c_fp)
        out = {}
        for j in cand:
            rs = []
            for v in self._outcomes(row, j):
                r2 = row.copy()
                r2[j] = v
                rs.append(bayes_risk(self._predict1(r2), self.c_fn, self.c_fp))
            out[j] = r0 - float(np.mean(rs)) - self.lam * self.costs.get(j, 1.0)
        return p0, r0, out

    def step(self, row: pd.DataFrame, known: list):
        cand = [c for c in row.columns if c not in known]
        p0 = self._predict1(row)
        if not cand:
            return None, p0, {}
        _, _, vois = self.voi(row, known, cand)
        j = max(vois, key=vois.get)
        return (None, p0, vois) if vois[j] <= 0 else (j, p0, vois)
