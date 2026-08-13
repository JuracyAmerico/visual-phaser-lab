# -*- coding: utf-8 -*-
"""
Turning a model score into a number a genealogist can act on.

A posterior is only useful if it means what it says. Measured on simulated
families, the raw segment-level posterior was consistently *under*-confident:
segments the model scored 0.61 were right 76% of the time, and segments it
scored 0.78 were right 92%. Under-confidence is the safe direction to be wrong
in, but it is still wrong — a researcher who discards a correct call because
the tool said 0.6 has lost real evidence.

The gap has a mechanical cause. A segment call is a majority vote across many
markers, and a vote among many correlated-but-not-identical estimates is more
reliable than the average of their individual confidences. So the raw score is
a monotone but miscalibrated function of the truth, which is exactly the
situation isotonic regression is for: fit the monotone map from score to
observed accuracy, and report the mapped value.

The fit is only honest if it is validated out of sample, so `evaluate()` scores
a calibrator on data it never saw. A calibrator fitted and reported on the same
families would be a tautology.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def pool_adjacent_violators(y, w):
    """
    Isotonic regression by PAVA: the closest non-decreasing fit to `y`.

    Written out rather than pulled from scipy because this repo deliberately
    depends only on numpy and pandas, and the algorithm is fifteen lines.
    """
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()
    n = len(y)
    values, weights, sizes = [], [], []
    for i in range(n):
        values.append(y[i])
        weights.append(w[i])
        sizes.append(1)
        # Merge backwards while the sequence decreases.
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, s2 = values.pop(), weights.pop(), sizes.pop()
            v1, w1, s1 = values.pop(), weights.pop(), sizes.pop()
            total = w1 + w2
            values.append((v1 * w1 + v2 * w2) / total)
            weights.append(total)
            sizes.append(s1 + s2)
    out = np.empty(n)
    at = 0
    for v, s in zip(values, sizes):
        out[at:at + s] = v
        at += s
    return out


@dataclass
class IsotonicCalibrator:
    """Monotone map from raw score to probability of being correct."""

    x: np.ndarray            # sorted knot scores
    y: np.ndarray            # fitted accuracy at each knot
    n_fit: int = 0
    note: str = ""

    @classmethod
    def fit(cls, scores, correct, n_knots=200, note=""):
        scores = np.asarray(scores, dtype=float)
        correct = np.asarray(correct, dtype=float)
        order = np.argsort(scores, kind="mergesort")
        scores, correct = scores[order], correct[order]

        # Bin before fitting: raw PAVA on tens of thousands of 0/1 outcomes
        # produces a step function with as many knots as distinct scores, which
        # overfits the tail where data is thin.
        edges = np.quantile(scores, np.linspace(0, 1, n_knots + 1))
        edges = np.unique(edges)
        which = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)
        xs, ys, ws = [], [], []
        for b in range(len(edges) - 1):
            sel = which == b
            if not sel.any():
                continue
            xs.append(float(scores[sel].mean()))
            ys.append(float(correct[sel].mean()))
            ws.append(float(sel.sum()))
        fitted = pool_adjacent_violators(ys, ws)
        return cls(x=np.asarray(xs), y=np.clip(fitted, 0.0, 1.0),
                   n_fit=len(scores), note=note)

    def transform(self, scores):
        """Calibrated probability for each raw score."""
        scores = np.asarray(scores, dtype=float)
        if len(self.x) == 0:
            return scores
        return np.interp(scores, self.x, self.y)

    def to_json(self, path):
        Path(path).write_text(json.dumps({
            "x": self.x.tolist(), "y": self.y.tolist(),
            "n_fit": int(self.n_fit), "note": self.note,
        }, indent=2))

    @classmethod
    def from_json(cls, path):
        blob = json.loads(Path(path).read_text())
        return cls(x=np.asarray(blob["x"]), y=np.asarray(blob["y"]),
                   n_fit=blob.get("n_fit", 0), note=blob.get("note", ""))


def expected_calibration_error(probabilities, correct, bins=10):
    """
    Mean |claimed probability - observed accuracy|, weighted by bin count.

    Equal-count bins rather than equal-width: with equal-width bins most of the
    mass lands in one or two bins and the statistic stops being informative.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(probabilities) == 0:
        return float("nan")
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(probabilities, edges[1:-1]), 0, len(edges) - 2)
    total, error = 0.0, 0.0
    for b in range(len(edges) - 1):
        sel = which == b
        n = int(sel.sum())
        if n == 0:
            continue
        error += n * abs(probabilities[sel].mean() - correct[sel].mean())
        total += n
    return float(error / total) if total else float("nan")


def evaluate(calibrator, scores, correct, bins=10):
    """Out-of-sample calibration error, before and after recalibration."""
    return {
        "n": int(len(scores)),
        "ece_raw": round(expected_calibration_error(scores, correct, bins), 4),
        "ece_calibrated": round(
            expected_calibration_error(calibrator.transform(scores), correct, bins), 4),
        "accuracy": round(float(np.mean(correct)), 4),
        "mean_raw": round(float(np.mean(scores)), 4),
        "mean_calibrated": round(float(np.mean(calibrator.transform(scores))), 4),
    }
