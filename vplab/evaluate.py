# -*- coding: utf-8 -*-
"""
Measurement harness — what the phasing model actually gets right.

Every accuracy figure this project publishes has to come from here, run on
demand, rather than from a number someone remembers. That rule exists because
the first round of figures did not survive re-measurement:

  * "85.5% accurate" was per *marker*. The application reads *segments*, and
    segment accuracy was 66.5%.
  * "the simulator is pessimistic, it has no LD" was backwards. The simulator
    drew a private reference/alternate pair per founder, so unrelated people
    almost never shared an allele and exclusions — the strongest signal the
    method has — arrived at ~17x their real rate. Simulated data was *easier*
    than real data, and the accuracy measured on it was an overstatement.

Three things are measured here, all against known truth:

  partition accuracy   which siblings share a grandparental haplotype, per
                       marker and per segment, compared label-invariantly
  crossover recovery   count and position of inferred switches per side, which
                       is where a wrong transition model shows up first
  calibration          reported posterior vs. observed accuracy. A model that
                       says 0.9 and is right 0.65 of the time is worse than one
                       that says 0.65, because a genealogist acts on the number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import phase_hmm as H
from . import simulate as S

FIR, HIR, NIR, MISSING = H.FIR, H.HIR, H.NIR, H.MISSING
NO_CALL_CHARS = ("-", "0", "X", "I", "D")


# ----------------------------------------------------------------------
# Pairwise match calls
# ----------------------------------------------------------------------
def match_calls(geno_a, geno_b):
    """
    Per-marker IBS class between two genotype frames.

    Deliberately mirrors the corrected upstream engine: a no-call anywhere in
    either genotype yields MISSING, which is an absence of observation rather
    than a state, and must not count for or against any hypothesis.
    """
    a1 = geno_a["allele1"].values.astype("U1")
    a2 = geno_a["allele2"].values.astype("U1")
    b1 = geno_b["allele1"].values.astype("U1")
    b2 = geno_b["allele2"].values.astype("U1")

    missing = np.zeros(len(a1), dtype=bool)
    for arr in (a1, a2, b1, b2):
        for token in NO_CALL_CHARS:
            missing |= arr == token

    fir = ((a1 == b1) & (a2 == b2)) | ((a1 == b2) & (a2 == b1))
    shares = (a1 == b1) | (a1 == b2) | (a2 == b1) | (a2 == b2)

    out = np.full(len(a1), HIR, dtype=object)
    out[fir] = FIR
    out[~shares] = NIR
    out[missing] = MISSING
    return out


# ----------------------------------------------------------------------
# Label-invariant partitions
# ----------------------------------------------------------------------
def canonical_partition(bits):
    """
    Canonical form of a binary vector under global flip.

    Which grandparent is called '0' is arbitrary — only *which siblings match
    each other* is recoverable without an outside anchor. Forcing the first
    entry to 0 makes two labellings of the same partition compare equal.
    """
    bits = np.asarray(bits)
    if bits.ndim == 1:
        return tuple((bits ^ bits[0]).tolist())
    return [tuple(row.tolist()) for row in (bits ^ bits[:, :1])]


def truth_bits(pedigree, siblings, side, marker_index=None):
    """
    Ground-truth inheritance bits: (markers, siblings), 0/1 per grandparent.
    """
    labels = np.stack(
        [pedigree.origins[name][side].values for name in siblings], axis=1
    )
    if marker_index is not None:
        labels = labels[marker_index]
    first_label = labels[0, 0]
    reference = {
        "paternal": ("PGF", "PGM"),
        "maternal": ("MGF", "MGM"),
    }[side][0]
    _ = first_label  # kept explicit: labelling is by grandparent, not by order
    return (labels != reference).astype(np.int8)


# ----------------------------------------------------------------------
# Running the model
# ----------------------------------------------------------------------
@dataclass
class ChromosomeResult:
    chromosome: int
    keep: np.ndarray                  # indices into the full marker table
    cm: np.ndarray
    path: np.ndarray
    states: list
    posterior: np.ndarray             # raw, (markers, states)
    n_siblings: int

    def partition_posterior(self, side_index):
        """P(partition | data) per marker on one side. See phase_hmm."""
        return H.partition_posterior(self.posterior, self.states, side_index)

    def called_partition(self, side_index):
        """Viterbi partition per marker, canonicalised."""
        bits = np.array([self.states[s][side_index] for s in self.path])
        return canonical_partition(bits)


def calibrate_from_pedigree(pedigree, sibling="Sib1"):
    """
    Emission model from the simulated family's own calibration relatives.

    The unrelated individual gives IBD0 and a parent gives IBD1 by definition,
    exactly as on real data — so the evaluation never assumes the emission
    table it is testing.
    """
    unrelated = match_calls(pedigree.relatives["Unrelated"], pedigree.genotypes[sibling])
    parent = match_calls(pedigree.relatives["Father"], pedigree.genotypes[sibling])
    return H.calibrate_emissions(unrelated, parent, min_count=1000)


def phase_pedigree(pedigree, emissions=None, thinning_cm=H.DEFAULT_THINNING_CM,
                   chromosomes=None, siblings=None, **hmm_kwargs):
    """Run the HMM over every chromosome of a simulated pedigree."""
    siblings = siblings or pedigree.siblings
    emissions = emissions or calibrate_from_pedigree(pedigree)
    n = len(siblings)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    calls = np.stack(
        [match_calls(pedigree.genotypes[siblings[i]], pedigree.genotypes[siblings[j]])
         for i, j in pairs]
    )

    markers = pedigree.markers
    results = []
    for chrom in sorted(markers["chromosome"].unique()):
        if chromosomes is not None and chrom not in chromosomes:
            continue
        rows = np.flatnonzero((markers["chromosome"] == chrom).values)
        cm_all = markers["cm"].values[rows]
        keep_local = H.thin_markers(cm_all, thinning_cm)
        keep = rows[keep_local]
        cm = markers["cm"].values[keep]

        # Drive each side by its own map when the panel carries one.
        sex_cm = {}
        if {"cm_male", "cm_female"} <= set(markers.columns):
            sex_cm = {
                "cm_male": markers["cm_male"].values[keep],
                "cm_female": markers["cm_female"].values[keep],
            }

        path, posterior, states = H.phase_chromosome(
            calls[:, keep], pairs, n, cm, emissions, **sex_cm, **hmm_kwargs
        )
        results.append(ChromosomeResult(
            chromosome=int(chrom), keep=keep, cm=cm, path=path,
            states=states, posterior=posterior, n_siblings=n,
        ))
    return results, emissions, siblings


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
SIDES = (("paternal", 0), ("maternal", 1))


def marker_accuracy(pedigree, results, siblings):
    """Per-marker partition accuracy, and the posterior attached to each call."""
    rows = []
    for res in results:
        for side, idx in SIDES:
            truth = canonical_partition(truth_bits(pedigree, siblings, side, res.keep))
            called = res.called_partition(idx)
            keys, post = res.partition_posterior(idx)
            key_index = {k: i for i, k in enumerate(keys)}
            confidence = np.array([
                post[t, key_index[called[t]]] for t in range(len(called))
            ])
            rows.append(pd.DataFrame({
                "chromosome": res.chromosome,
                "side": side,
                "cm": res.cm,
                "correct": [c == t for c, t in zip(called, truth)],
                "posterior": confidence,
            }))
    return pd.concat(rows, ignore_index=True)


def segment_accuracy(pedigree, results, siblings, n_windows=400,
                     span_cm=(10.0, 30.0), seed=0):
    """
    Segment-level partition accuracy — the number the application depends on.

    A genealogical match arrives as an interval of some tens of cM, and the
    question asked of it is "which grandparental line". So the unit of
    measurement is a window, not a marker, and the call is the majority
    assignment across it.

    Windows spanning a true crossover are skipped: there the question has no
    single right answer, and scoring them would measure the wrong thing in
    either direction.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for res in results:
        for side, idx in SIDES:
            truth = canonical_partition(truth_bits(pedigree, siblings, side, res.keep))
            called = res.called_partition(idx)
            keys, post = res.partition_posterior(idx)
            key_index = {k: i for i, k in enumerate(keys)}
            lo, hi = res.cm[0], res.cm[-1]
            for _ in range(n_windows):
                width = rng.uniform(*span_cm)
                if hi - lo <= width:
                    continue
                start = rng.uniform(lo, hi - width)
                sel = np.flatnonzero((res.cm >= start) & (res.cm <= start + width))
                if len(sel) < 10:
                    continue
                window_truth = {truth[t] for t in sel}
                if len(window_truth) != 1:
                    continue                     # crossover inside: ill-posed
                truth_key = window_truth.pop()

                counts = {}
                for t in sel:
                    counts[called[t]] = counts.get(called[t], 0) + 1
                majority = max(counts, key=counts.get)
                mean_post = float(np.mean([
                    post[t, key_index[majority]] for t in sel
                ]))
                rows.append({
                    "chromosome": res.chromosome,
                    "side": side,
                    "start_cm": round(start, 2),
                    "span_cm": round(width, 2),
                    "markers": len(sel),
                    "correct": majority == truth_key,
                    "posterior": mean_post,
                })
    return pd.DataFrame(rows)


def crossover_counts(pedigree, results, siblings):
    """
    Inferred vs. true crossover counts per side.

    The most diagnostic single number in the whole model. Human meiosis is
    tightly constrained — roughly 26 crossovers per male gamete and 42 per
    female — so a systematic deficit or excess on one side points straight at
    the transition model rather than at the data.
    """
    rows = []
    for res in results:
        for side, idx in SIDES:
            bits = np.array([res.states[s][idx] for s in res.path])
            for k, name in enumerate(siblings):
                inferred = int(np.count_nonzero(np.diff(bits[:, k])))
                true_co = pedigree.crossovers[name][side].get(res.chromosome, [])
                rows.append({
                    "chromosome": res.chromosome,
                    "side": side,
                    "sibling": name,
                    "inferred": inferred,
                    "true": len(true_co),
                    "cm_span": round(float(res.cm[-1] - res.cm[0]), 1),
                })
    return pd.DataFrame(rows)


def calibration_curve(frame, bins=None, min_count=20):
    """
    Reported posterior vs. observed accuracy.

    The plot no tool in this field publishes. Perfect calibration is the
    diagonal; anything above it is a model that understates its confidence,
    anything below is one that lies to the user.
    """
    bins = bins if bins is not None else np.array([0.25, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0001])
    which = np.digitize(frame["posterior"].values, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        sel = which == b
        n = int(sel.sum())
        if n < min_count:
            continue
        rows.append({
            "posterior_low": bins[b],
            "posterior_high": min(bins[b + 1], 1.0),
            "n": n,
            "mean_posterior": round(float(frame["posterior"].values[sel].mean()), 4),
            "observed_accuracy": round(float(frame["correct"].values[sel].mean()), 4),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["gap"] = (out["mean_posterior"] - out["observed_accuracy"]).round(4)
    return out


def calibration_error(curve):
    """
    Expected calibration error: mean |posterior - accuracy|, weighted by count.
    """
    if curve.empty:
        return float("nan")
    w = curve["n"].values / curve["n"].values.sum()
    return float(np.sum(w * np.abs(curve["gap"].values)))
