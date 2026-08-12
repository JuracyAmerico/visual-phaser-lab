# -*- coding: utf-8 -*-
"""
Trio consistency — the transitivity constraint on sibling comparisons.

At a locus, sibling i has a paternal grandparental origin p_i in {PGF, PGM} and
a maternal origin m_i in {MGF, MGM}. For a pair:

    FIR  iff  p_i == p_j  and  m_i == m_j
    NIR  iff  p_i != p_j  and  m_i != m_j
    HIR  otherwise

Equality on a two-valued attribute is transitive, so across the three pairs of
three siblings the number of paternal matches is 3 (all same) or 1 (two same,
one different) -- never 0, never 2. The same holds maternally. Enumerating the
four combinations yields every observable outcome:

    paternal  maternal   (FIR, HIR, NIR)
       3         3          (3, 0, 0)
       3         1          (1, 2, 0)
       1         3          (1, 2, 0)
       1         1 (same pair)      (1, 0, 2)
       1         1 (different pair) (0, 2, 1)

Everything else is impossible under the assumption that the three siblings are
full siblings sharing the same four grandparents. This is repository issue #4,
and it is decidable in O(1) per locus.

Its real value is as an oracle: a violation means at least one of the three
pairwise calls at that locus is wrong. That makes the violation rate an
objective quality measure on real data with no ground truth required, and its
residual level is an estimate of the genotype error rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The tool encodes genetic state as display colours.
FIR, HIR, NIR = "limegreen", "yellow", "crimson"

# (n_fir, n_hir, n_nir) combinations reachable by inheritance.
VALID_COMBINATIONS = {(3, 0, 0), (1, 2, 0), (1, 0, 2), (0, 2, 1)}

# Human-readable interpretation of each valid combination.
COMBINATION_MEANING = {
    (3, 0, 0): "all three siblings share both grandparental copies",
    (1, 2, 0): "one pair fully identical, other two share one copy",
    (1, 0, 2): "one pair fully identical, other two share nothing",
    (0, 2, 1): "two pairs share one copy, third pair shares nothing",
}


def check_trio(match_ab, match_ac, match_bc, positions=None):
    """
    Vectorised consistency check over three aligned pairwise match arrays.

    Returns a DataFrame with one row per locus: the three calls, the
    (FIR, HIR, NIR) counts, and whether the combination is reachable.
    """
    a, b, c = np.asarray(match_ab), np.asarray(match_ac), np.asarray(match_bc)
    if not (len(a) == len(b) == len(c)):
        raise ValueError("the three pairwise arrays must be aligned and equal length")

    stacked = np.vstack([a, b, c])
    n_fir = (stacked == FIR).sum(axis=0)
    n_hir = (stacked == HIR).sum(axis=0)
    n_nir = (stacked == NIR).sum(axis=0)

    # Loci where any pair lacks data are not evaluable.
    evaluable = (n_fir + n_hir + n_nir) == 3

    valid = np.zeros(len(a), dtype=bool)
    for f, h, n in VALID_COMBINATIONS:
        valid |= (n_fir == f) & (n_hir == h) & (n_nir == n)

    frame = pd.DataFrame(
        {
            "AB": a, "AC": b, "BC": c,
            "n_fir": n_fir, "n_hir": n_hir, "n_nir": n_nir,
            "evaluable": evaluable,
            "violation": evaluable & ~valid,
        }
    )
    if positions is not None:
        frame.insert(0, "position", np.asarray(positions))
    return frame


def violation_summary(frame):
    """Aggregate a check_trio result into a reportable summary."""
    evaluable = frame["evaluable"].sum()
    violations = frame["violation"].sum()
    rate = violations / evaluable if evaluable else float("nan")

    breakdown = (
        frame[frame["violation"]]
        .groupby(["n_fir", "n_hir", "n_nir"])
        .size()
        .sort_values(ascending=False)
    )
    return {
        "loci": len(frame),
        "evaluable": int(evaluable),
        "violations": int(violations),
        "violation_rate": rate,
        "breakdown": breakdown,
    }


def describe_combination(n_fir, n_hir, n_nir):
    """Explain a combination, or say why it cannot occur."""
    key = (n_fir, n_hir, n_nir)
    if key in COMBINATION_MEANING:
        return COMBINATION_MEANING[key]
    reasons = []
    if n_hir == 3:
        reasons.append("three HIR requires exactly two paternal matches, but equality is transitive")
    if n_nir == 3:
        reasons.append("three NIR requires zero paternal matches, impossible among three siblings")
    if n_fir == 2:
        reasons.append("two FIR forces the third pair to be FIR as well")
    if n_fir == 1 and n_hir == 1 and n_nir == 1:
        reasons.append("one of each is unreachable from any grandparental assignment")
    return "; ".join(reasons) or "unreachable combination"
