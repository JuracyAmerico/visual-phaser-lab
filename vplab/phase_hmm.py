# -*- coding: utf-8 -*-
"""
Grandparental assignment by inheritance-vector HMM.

Visual phasing is a special case of the Lander-Green algorithm: model pedigree
transmission as a hidden Markov chain whose state is an *inheritance vector*
recording which parental homolog each child received at that locus. Transitions
between adjacent markers follow the recombination fraction, which comes
straight from the genetic map. Merlin has done this for linkage analysis for
decades; nobody has pointed it at consumer DNA for genealogy.

What it buys over reading the plot by eye:

  * a MAP assignment of every segment to a grandparental group (Viterbi)
  * a posterior probability per segment (forward-backward) instead of a colour
  * crossover positions with uncertainty intervals
  * N siblings rather than exactly three -- every existing visual-phasing aid
    is built for three because that is a human limit, not an algorithmic one

STATE SPACE
    2^(2n) states: 64 for three siblings, 1024 for five. Trivial at this scale.

    The textbook reduction pins sibling 0 to (0, 0) to quotient out the
    arbitrary labelling of the two grandparents per side, cutting the space
    fourfold. That is avoided here on purpose. Pinning makes every state
    relative to sibling 0, so that sibling's own crossovers reappear as
    simultaneous switches in all the others and the per-sibling output shows
    them recombining nowhere -- an artefact that looks entirely plausible until
    you notice one person has a single unbroken segment per chromosome.

    The residual degeneracy (a state and its global per-side flip describe the
    same partition) is collapsed when posteriors are reported, so confidence
    reflects the partition rather than being split across indistinguishable
    labellings.

    The labels recovered are a *partition*, not names. Which group is the
    paternal grandfather rather than the paternal grandmother needs an outside
    anchor -- a tested cousin, aunt or great-aunt on a known line.

EMISSIONS
    Calibrated from the data rather than assumed. Identity-by-state coincidence
    -- two people matching without inheriting from a common ancestor -- is the
    dominant source of noise, and its rate depends on the marker panel and the
    population. Two pairs in a typical family dataset pin it exactly:

        IBD0 distribution  <- an unrelated pair (share nothing anywhere)
        IBD1 distribution  <- a parent-child pair (share exactly one copy
                              at every locus, by definition)
        IBD2 distribution  <- FIR, with the error rate taken from apparent
                              exclusions in that same parent-child pair, which
                              cannot be real and must be genotyping error

    This is what makes the model honest: nothing is borrowed from a paper about
    a different chip and a different population.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

FIR, HIR, NIR, MISSING = "limegreen", "yellow", "crimson", "grey"
OBS = (FIR, HIR, NIR)

# Female meiosis recombines more than male; the map here is sex-averaged, so
# each side is scaled toward its own rate. Genome-wide averages are ~26
# crossovers per male gamete and ~42 per female.
#
# WARNING: these scales are a PRIOR IMPOSED ON side 0 and side 1, not something
# the model discovers. Side 0 is told to recombine less. Observing that it does
# is therefore circular, and crossover counts CANNOT be used to decide which
# side is the father's. Identifying that requires an external anchor -- a
# relative known to descend from one specific grandparent.
MALE_SCALE, FEMALE_SCALE = 0.78, 1.25

# Default marker spacing, in cM, for thin_markers(). Calibrated on real data by
# sweeping the spacing until inferred crossover counts matched the biological
# rate: unthinned gave 234 paternal / 872 maternal across three siblings where
# ~78 / ~126 are expected; 0.2 cM gave 44 / 128.
DEFAULT_THINNING_CM = 0.2


@dataclass
class EmissionModel:
    """P(observed IBS state | true IBD state), IBD in {0, 1, 2}."""

    table: np.ndarray          # shape (3 IBD, 3 OBS)
    error_rate: float
    source: str

    def loglik(self, ibd, obs_index):
        return np.log(self.table[ibd, obs_index] + 1e-12)


def calibrate_emissions(unrelated_pair, parent_child_pair, min_count=10000):
    """
    Estimate P(observed | IBD) from pairs whose IBD state is known a priori.

    `unrelated_pair` and `parent_child_pair` are arrays of match calls
    ('limegreen'/'yellow'/'crimson'/'grey') for one pair each, genome-wide.

    An unrelated pair is IBD0 everywhere. A parent and child are IBD1
    everywhere -- they share exactly one copy at every locus, no more and no
    less. So the observed distributions in those two pairs *are* the emission
    rows, with no modelling assumption at all.
    """
    def distribution(calls):
        calls = np.asarray(calls)
        calls = calls[calls != MISSING]
        if len(calls) < min_count:
            raise ValueError(f"only {len(calls)} informative markers; need {min_count}")
        counts = np.array([(calls == state).sum() for state in OBS], dtype=float)
        return counts / counts.sum(), len(calls)

    ibd0, n0 = distribution(unrelated_pair)
    ibd1, n1 = distribution(parent_child_pair)

    # A parent and child cannot be genuinely excluded anywhere. Any NIR call in
    # that pair is therefore a genotyping error, which gives the error rate
    # directly.
    error_rate = float(ibd1[OBS.index(NIR)])

    # IBD2: fully identical, degraded by the same error rate. An error can push
    # it to HIR; two coincident errors to NIR, which is negligible.
    ibd2 = np.array([1.0 - error_rate, error_rate * 0.95, error_rate * 0.05])
    ibd2 /= ibd2.sum()

    return EmissionModel(
        table=np.vstack([ibd0, ibd1, ibd2]),
        error_rate=error_rate,
        source=f"IBD0 from {n0:,} unrelated markers; IBD1 from {n1:,} parent-child markers",
    )


def _states(n_siblings):
    """
    All inheritance vectors: 2^(2n) of them.

    Sibling 0 is deliberately NOT pinned. Pinning it is the usual symmetry
    reduction and it halves the state space on each side, but it makes the
    state relative to sibling 0 -- so that sibling's own crossovers get
    absorbed into simultaneous switches of everyone else, and the per-sibling
    output shows them as never recombining. That is an artefact, and a
    convincing-looking one.

    The cost of keeping the full space is a factor of four (64 states for three
    siblings instead of 16), which is irrelevant at this scale. The label
    degeneracy that remains -- a state and its global flip describe the same
    partition -- is handled when posteriors are reported, not here.
    """
    return [
        (np.array(p, dtype=np.int8), np.array(m, dtype=np.int8))
        for p in product((0, 1), repeat=n_siblings)
        for m in product((0, 1), repeat=n_siblings)
    ]


def _pair_ibd(states, pairs):
    """
    IBD count (0/1/2) for each (state, pair). Two siblings are IBD1 on the
    paternal side exactly when they received the same paternal homolog.
    """
    out = np.zeros((len(states), len(pairs)), dtype=np.int8)
    for s, (pat, mat) in enumerate(states):
        for k, (i, j) in enumerate(pairs):
            out[s, k] = int(pat[i] == pat[j]) + int(mat[i] == mat[j])
    return out


def _hamming_matrices(states):
    """
    Per-side Hamming distance between every pair of states.

    These do not depend on the recombination fraction, so they are computed
    once and reused for every marker. Rebuilding them per marker in a Python
    double loop is what made the first version unusable: 64 states means 4096
    inner iterations per marker, times ~600k markers.
    """
    pat = np.array([p for p, _ in states])
    mat = np.array([m for _, m in states])
    flips_p = (pat[:, None, :] != pat[None, :, :]).sum(axis=2)
    flips_m = (mat[:, None, :] != mat[None, :, :]).sum(axis=2)
    return flips_p, flips_m


def _transition_logp(flips_p, flips_m, n_siblings, theta_pat, theta_mat):
    """
    Log transition matrix for one inter-marker interval.

    Each meiosis recombines independently, so the transition factorises over
    siblings: flipping k paternal bits costs theta_pat^k (1-theta_pat)^(n-k),
    and likewise maternally. Fully vectorised over the precomputed Hamming
    distances.
    """
    tp = float(np.clip(theta_pat, 1e-9, 0.4999))
    tm = float(np.clip(theta_mat, 1e-9, 0.4999))
    return (
        flips_p * np.log(tp) + (n_siblings - flips_p) * np.log1p(-tp)
        + flips_m * np.log(tm) + (n_siblings - flips_m) * np.log1p(-tm)
    )


def _haldane(delta_cm, scale):
    """Map distance to recombination fraction (Haldane, no interference)."""
    d = np.maximum(delta_cm, 0.0) * scale / 100.0
    return 0.5 * (1.0 - np.exp(-2.0 * d))


def phase_chromosome(match_by_pair, pairs, n_siblings, cm, emissions):
    """
    Run forward-backward and Viterbi over one chromosome.

    `match_by_pair` is (n_pairs, n_markers) of match calls; `cm` is the genetic
    position of each marker. Returns the MAP state path and the per-marker
    posterior over states.
    """
    states = _states(n_siblings)
    n_states, n_markers = len(states), len(cm)
    pair_ibd = _pair_ibd(states, pairs)
    flips_p, flips_m = _hamming_matrices(states)

    # Transition depends on the interval only through its genetic length, and
    # adjacent-marker spacings repeat heavily once rounded. Cache on that.
    trans_cache = {}
    def transition(d):
        key = round(float(d), 5)
        if key not in trans_cache:
            trans_cache[key] = _transition_logp(
                flips_p, flips_m, n_siblings,
                _haldane(key, MALE_SCALE), _haldane(key, FEMALE_SCALE))
        return trans_cache[key]

    # Emission log-likelihood per (marker, state), summed over pairs. Missing
    # calls contribute nothing, which is the correct handling of absent data:
    # it neither supports nor refutes any state.
    obs_index = np.full(match_by_pair.shape, -1, dtype=np.int8)
    for k, state in enumerate(OBS):
        obs_index[match_by_pair == state] = k

    log_e = np.zeros((n_markers, n_states))
    for s in range(n_states):
        acc = np.zeros(n_markers)
        for p in range(len(pairs)):
            seen = obs_index[p] >= 0
            acc[seen] += emissions.loglik(pair_ibd[s, p], obs_index[p][seen])
        log_e[:, s] = acc

    delta = np.diff(cm, prepend=cm[0])
    log_prior = -np.log(n_states)

    # --- Viterbi -------------------------------------------------------
    delta_v = np.full((n_markers, n_states), -np.inf)
    back = np.zeros((n_markers, n_states), dtype=np.int32)
    delta_v[0] = log_prior + log_e[0]
    for t in range(1, n_markers):
        scores = delta_v[t - 1][:, None] + transition(delta[t])
        back[t] = np.argmax(scores, axis=0)
        delta_v[t] = scores[back[t], np.arange(n_states)] + log_e[t]

    path = np.zeros(n_markers, dtype=np.int32)
    path[-1] = int(np.argmax(delta_v[-1]))
    for t in range(n_markers - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    # --- Forward-backward ----------------------------------------------
    def logsumexp(a, axis=None):
        m = np.max(a, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)

    fwd = np.full((n_markers, n_states), -np.inf)
    fwd[0] = log_prior + log_e[0]
    for t in range(1, n_markers):
        fwd[t] = logsumexp(fwd[t - 1][:, None] + transition(delta[t]), axis=0) + log_e[t]

    bwd = np.full((n_markers, n_states), -np.inf)
    bwd[-1] = 0.0
    for t in range(n_markers - 2, -1, -1):
        bwd[t] = logsumexp(transition(delta[t + 1]) + log_e[t + 1] + bwd[t + 1], axis=1)

    post = fwd + bwd
    post -= logsumexp(post, axis=1)[:, None]
    post = np.exp(post)

    # A state and its global per-side flip describe the same partition with the
    # grandparental labels swapped. Report the probability of the *partition*,
    # summed over its equivalent labellings, or confidence looks artificially
    # low simply because it is split across indistinguishable states.
    index = {(tuple(p), tuple(m)): i for i, (p, m) in enumerate(states)}
    groups = {}
    for i, (p, m) in enumerate(states):
        key = min((tuple(p), tuple(m)), (tuple(1 - p), tuple(m)),
                  (tuple(p), tuple(1 - m)), (tuple(1 - p), tuple(1 - m)))
        groups.setdefault(key, []).append(i)
    collapsed = post.copy()
    for members in groups.values():
        collapsed[:, members] = post[:, members].sum(axis=1, keepdims=True)
    return path, collapsed, states


def assign_grandparents(path, states, positions, cm, n_siblings, min_cm=1.0):
    """
    Convert a state path into per-sibling grandparental segments.

    Group labels are arbitrary: 'P0'/'P1' are the two paternal grandparents and
    'M0'/'M1' the two maternal ones, but which real ancestor each corresponds to
    requires an external anchor.
    """
    rows = []
    for side, letter in ((0, "P"), (1, "M")):
        bits = np.array([states[s][side] for s in path])       # (markers, siblings)
        for sib in range(n_siblings):
            series = bits[:, sib]
            edges = np.flatnonzero(np.diff(series)) + 1
            starts = np.concatenate([[0], edges])
            ends = np.concatenate([edges, [len(series)]])
            for a, b in zip(starts, ends):
                length = cm[b - 1] - cm[a]
                if length < min_cm:
                    continue
                rows.append({
                    "sibling": sib,
                    "side": "paternal" if side == 0 else "maternal",
                    "grandparent_group": f"{letter}{int(series[a])}",
                    "start": int(positions[a]),
                    "end": int(positions[b - 1]),
                    "cM": round(float(length), 2),
                    "markers": int(b - a),
                })
    return pd.DataFrame(rows)


def thin_markers(cm, min_gap_cm):
    """
    Indices of a marker subset spaced at least `min_gap_cm` apart.

    The emission model treats markers as independent. Real markers are not:
    linkage disequilibrium correlates neighbours, so a dense panel supplies far
    less information than its count implies. The HMM, believing every marker,
    lets noise outvote the transition prior and switches state far too often --
    measured at 3-7x the biological crossover rate on real data, against ~1x on
    LD-free simulated data, which is what identified LD as the cause.

    Thinning to a spacing beyond the LD decay length restores the independence
    the model assumes. Segments of genealogical interest are >=7 cM, so even
    coarse thinning leaves tens of markers per segment.
    """
    keep = [0]
    last = cm[0]
    for i in range(1, len(cm)):
        if cm[i] - last >= min_gap_cm:
            keep.append(i)
            last = cm[i]
    return np.array(keep)
