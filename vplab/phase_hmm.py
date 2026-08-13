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

# Fallback global rate scales, used ONLY when no sex-specific map is available.
# They reproduce the genome-wide totals (2608 cM male, 4094 cM female against a
# 3351 cM sex-averaged map) and nothing else.
#
# WARNING: as a *scale*, this is a prior imposed on side 0 and side 1, not
# something the model discovers -- side 0 is simply told to recombine less. So
# crossover counts cannot be used to decide which side is the father's, and
# measured, the two sides came out transposed on 2 of 10 simulated chromosomes.
# Prefer real sex-specific maps (genmap.py), which differ in *shape* and make
# side identity locally identifiable; naming which grandparent is which still
# requires an outside anchor.
MALE_SCALE, FEMALE_SCALE = 0.78, 1.25

# Default marker spacing, in cM, for thin_markers().
#
# Chosen against an external biological constant rather than by eye: the
# inferred crossover count per gamete has to land near the ~26 male / ~41
# female that human meiosis actually produces. Swept over two independent real
# trios (ratios to literature, both sides):
#
#     unthinned  5.6-6.2x   linkage disequilibrium dominates completely
#     0.02 cM    0.92-1.16
#     0.08 cM    0.84-1.07
#     0.20 cM    0.70-1.02   the previous default, clearly over-thinned
#
# 0.05-0.10 is a plateau, and differences inside it are within the sampling
# noise of ~78 crossovers (about +/-11%). The old 0.2 cM value was compensating
# for a mis-specified transition model: with the sex-averaged map the chain
# over-switched, and heavy thinning hid it. With sex-specific maps that crutch
# is no longer needed, and keeping less of it leaves more information in.
DEFAULT_THINNING_CM = 0.08


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


def _haldane(delta_cm, scale=1.0):
    """Map distance to recombination fraction, assuming no interference."""
    d = np.maximum(delta_cm, 0.0) * scale / 100.0
    return 0.5 * (1.0 - np.exp(-2.0 * d))


def _kosambi(delta_cm, scale=1.0):
    """
    Map distance to recombination fraction under moderate positive
    interference.

    Real crossovers inhibit further crossovers nearby — two within a few cM are
    much rarer than independence predicts. A first-order Markov chain cannot
    represent that directly, because it has no memory of how far back the last
    switch was; augmenting the state with that distance would multiply an
    already 4^n state space.

    Kosambi's map function is the standard way to fold interference into the
    transition probability itself instead. It agrees with Haldane for short
    intervals and assigns systematically lower recombination fractions to long
    ones, which is what suppressing tight double crossovers looks like when
    expressed per interval.
    """
    d = np.maximum(delta_cm, 0.0) * scale / 100.0
    return 0.5 * np.tanh(2.0 * d)


MAP_FUNCTIONS = {"haldane": _haldane, "kosambi": _kosambi}


def phase_chromosome(match_by_pair, pairs, n_siblings, cm, emissions,
                     cm_male=None, cm_female=None, map_function="kosambi"):
    """
    Run forward-backward and Viterbi over one chromosome.

    `match_by_pair` is (n_pairs, n_markers) of match calls; `cm` is the
    sex-averaged genetic position of each marker.

    Pass `cm_male` and `cm_female` to drive each side by its own genetic map.
    That is strongly preferred: without them the two sides differ only by a
    global rate scale, and the labels which side is which become nearly
    unidentifiable within a single chromosome — measured at 2 of 10 simulated
    chromosomes coming out transposed.

    Returns the MAP state path, the per-marker posterior over states, and the
    state list.
    """
    states = _states(n_siblings)
    n_states, n_markers = len(states), len(cm)
    pair_ibd = _pair_ibd(states, pairs)
    flips_p, flips_m = _hamming_matrices(states)
    theta_of = MAP_FUNCTIONS[map_function]

    sex_specific = cm_male is not None and cm_female is not None
    if sex_specific:
        delta_pat = np.diff(np.asarray(cm_male, dtype=float), prepend=cm_male[0])
        delta_mat = np.diff(np.asarray(cm_female, dtype=float), prepend=cm_female[0])
        scale_pat = scale_mat = 1.0
    else:
        delta = np.diff(cm, prepend=cm[0])
        delta_pat = delta_mat = delta
        scale_pat, scale_mat = MALE_SCALE, FEMALE_SCALE

    # Transition depends on the interval only through its two genetic lengths,
    # and spacings repeat heavily once rounded. Cache on that.
    trans_cache = {}
    def transition(t):
        key = (round(float(delta_pat[t]), 5), round(float(delta_mat[t]), 5))
        if key not in trans_cache:
            trans_cache[key] = _transition_logp(
                flips_p, flips_m, n_siblings,
                theta_of(key[0], scale_pat), theta_of(key[1], scale_mat))
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

    log_prior = -np.log(n_states)

    # --- Viterbi -------------------------------------------------------
    delta_v = np.full((n_markers, n_states), -np.inf)
    back = np.zeros((n_markers, n_states), dtype=np.int32)
    delta_v[0] = log_prior + log_e[0]
    for t in range(1, n_markers):
        scores = delta_v[t - 1][:, None] + transition(t)
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
        fwd[t] = logsumexp(fwd[t - 1][:, None] + transition(t), axis=0) + log_e[t]

    bwd = np.full((n_markers, n_states), -np.inf)
    bwd[-1] = 0.0
    for t in range(n_markers - 2, -1, -1):
        bwd[t] = logsumexp(transition(t + 1) + log_e[t + 1] + bwd[t + 1], axis=1)

    post = fwd + bwd
    post -= logsumexp(post, axis=1)[:, None]
    post = np.exp(post)

    # The raw state posterior is returned deliberately. Collapsing label
    # degeneracy is a question about *which* claim you are making — the
    # per-side partition, the joint partition — so it belongs to the caller,
    # via partition_posterior() below, not to the inference.
    return path, post, states


def partition_posterior(posterior, states, side):
    """
    P(partition | data) per marker for one side (0 paternal, 1 maternal).

    Which grandparent is labelled '0' is not identifiable without an outside
    anchor; which siblings share a haplotype is. Marginalising onto the
    partition reports confidence in the claim actually being made. Reading
    confidence off the raw state posterior instead understates it, because the
    probability mass is split across labellings that are the same biological
    statement.

    Returns (keys, probabilities) where keys[k] is a canonical binary tuple
    with the first sibling fixed to 0, and probabilities is (markers, n_keys).
    """
    bits = np.array([s[side] for s in states])
    canon = [tuple((row ^ row[0]).tolist()) for row in bits]
    keys = sorted(set(canon))
    index = {k: i for i, k in enumerate(keys)}
    out = np.zeros((posterior.shape[0], len(keys)))
    for s, key in enumerate(canon):
        out[:, index[key]] += posterior[:, s]
    return keys, out


def assign_grandparents(path, states, positions, cm, n_siblings, min_cm=1.0,
                        posterior=None, calibrator=None, chromosome=None):
    """
    Convert a state path into per-sibling grandparental segments.

    Group labels are arbitrary: 'P0'/'P1' are the two paternal grandparents and
    'M0'/'M1' the two maternal ones, but which real ancestor each corresponds to
    requires an external anchor.

    Pass `posterior` to attach confidence to every segment. The confidence
    reported is the posterior of the *partition* — which siblings share a
    haplotype — because that is the identifiable claim. A per-sibling
    "probability this segment is P0" would be exactly 0.5 by construction: the
    model is perfectly symmetric under swapping the two grandparental labels,
    and no amount of data breaks that symmetry without an outside anchor.

    `calibrator` (a calibration.IsotonicCalibrator) maps the raw posterior onto
    observed accuracy; without one, the raw value is reported and is known to
    understate confidence.
    """
    rows = []
    partition_post = {}
    if posterior is not None:
        for side in (0, 1):
            keys, probs = partition_posterior(posterior, states, side)
            partition_post[side] = (keys, probs, {k: i for i, k in enumerate(keys)})

    for side, letter in ((0, "P"), (1, "M")):
        bits = np.array([states[s][side] for s in path])       # (markers, siblings)
        called = [tuple((row ^ row[0]).tolist()) for row in bits]
        for sib in range(n_siblings):
            series = bits[:, sib]
            edges = np.flatnonzero(np.diff(series)) + 1
            starts = np.concatenate([[0], edges])
            ends = np.concatenate([edges, [len(series)]])
            for a, b in zip(starts, ends):
                length = cm[b - 1] - cm[a]
                if length < min_cm:
                    continue
                row = {
                    "chromosome": chromosome,
                    "sibling": sib,
                    "side": "paternal" if side == 0 else "maternal",
                    "grandparent_group": f"{letter}{int(series[a])}",
                    "start": int(positions[a]),
                    "end": int(positions[b - 1]),
                    "cM": round(float(length), 2),
                    "markers": int(b - a),
                }
                if posterior is not None:
                    keys, probs, index = partition_post[side]
                    raw = float(np.mean([
                        probs[t, index[called[t]]] for t in range(a, b)
                    ]))
                    row["posterior"] = round(raw, 4)
                    row["confidence"] = round(
                        float(calibrator.transform([raw])[0]) if calibrator else raw, 4)
                rows.append(row)
    return pd.DataFrame(rows)


INSUFFICIENT = "INSUFFICIENT"


@dataclass
class SideCall:
    """The verdict on which grandparental line a match descends from."""

    call: str
    probability: float
    n_segments: int
    n_chromosomes: int
    support: float
    reason: str
    per_group: dict

    def __str__(self):
        if self.call == INSUFFICIENT:
            return f"{INSUFFICIENT} ({self.reason})"
        return (f"{self.call} p={self.probability:.2f} "
                f"[{self.n_segments} segments on {self.n_chromosomes} chromosomes]")


def assign_side(votes, min_chromosomes=3, n_groups=4):
    """
    Decide which grandparental group a match descends from, or decline to.

    `votes` is an iterable of (chromosome, group, probability_correct) — one
    entry per *discriminating* segment, meaning a segment where the siblings'
    matching pattern actually distinguishes the groups. A segment that all
    three siblings share equally says nothing and must not be passed in.

    Two rules, both bought with measurements rather than reasoning:

    **Votes on one chromosome are not independent.** Each chromosome is phased
    separately, and the failure that matters is a whole chromosome coming out
    with its two sides transposed — 2 of 10 simulated chromosomes did, before
    sex-specific maps. That mode flips every segment on the chromosome
    together, so ten segments from one chromosome are one piece of evidence,
    not ten. Votes are therefore collapsed per chromosome, keeping the strongest.

    **Fewer than three chromosomes is not a verdict.** At a measured ~86%
    per-segment accuracy a single segment still looks perfectly clean, because
    a single segment cannot contradict itself. That is precisely how a set of
    confident, mutually inconsistent line assignments got produced once here.
    Below the floor the function returns INSUFFICIENT and still reports what it
    computed, so the user can see how close the call was.

    Combination is Naive Bayes across chromosomes: a vote is right with its own
    probability and, when wrong, points at one of the other groups uniformly.
    That yields a genuinely calibrated number provided the per-segment
    probabilities are calibrated — which is what calibration.py is for.
    """
    votes = list(votes)
    if not votes:
        return SideCall(INSUFFICIENT, 0.0, 0, 0, 0.0, "no discriminating segments", {})

    strongest = {}
    for chromosome, group, probability in votes:
        probability = float(np.clip(probability, 1.0 / n_groups + 1e-9, 1 - 1e-9))
        current = strongest.get(chromosome)
        if current is None or probability > current[1]:
            strongest[chromosome] = (group, probability)

    groups = sorted({group for group, _ in strongest.values()})
    log_p = {g: 0.0 for g in groups}
    for group, probability in strongest.values():
        wrong = (1.0 - probability) / (n_groups - 1)
        for g in groups:
            log_p[g] += np.log(probability if g == group else wrong)

    top = max(log_p.values())
    weights = {g: float(np.exp(v - top)) for g, v in log_p.items()}
    total = sum(weights.values())
    posterior = {g: w / total for g, w in weights.items()}
    best = max(posterior, key=posterior.get)

    n_chromosomes = len(strongest)
    agreeing = sum(1 for group, _ in strongest.values() if group == best)
    support = agreeing / n_chromosomes

    if n_chromosomes < min_chromosomes:
        return SideCall(
            INSUFFICIENT, float(posterior[best]), len(votes), n_chromosomes, support,
            f"{n_chromosomes} chromosome(s) with discriminating segments, "
            f"{min_chromosomes} required", posterior,
        )
    return SideCall(best, float(posterior[best]), len(votes), n_chromosomes,
                    support, "", posterior)


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
