# -*- coding: utf-8 -*-
"""
Assign DNA matches to a parental side, using phased siblings.

    python -m tools.assign_matches --segments matches.csv \
        --siblings A B C --unrelated A M --parent-child A CHILD

The input CSV is one row per shared segment from a one-to-one comparison:
`match, sibling, chromosome, start, end, cM`. A sibling with no row over a
region did not match there.

HOW A SEGMENT DECIDES A SIDE
    A match shares a segment because both they and some of the siblings
    inherited it from one common ancestor. So the *set of siblings who match*
    over that segment is exactly the set who inherited that ancestor's
    haplotype there — which is one half of the phased partition.

    At every locus the model gives two partitions of the siblings, one per
    parental side. A segment is *discriminating* when the observed matching
    pattern is one half of exactly one of them: then only that side can explain
    it. When both sides would explain it (typically when everyone matches, or
    when both partitions happen to split the same way) the segment says nothing
    and is not counted.

WHAT THE ANSWER IS AND IS NOT
    "Which side" means which of the trio's two parents the match descends
    through — for a trio of siblings, their father's line or their mother's.
    That is a genome-wide claim and is what sex-specific maps made reliable.

    Which of the two grandparents *within* a side is a separate question this
    cannot answer: the labels 0 and 1 inside a side are arbitrary and, worse,
    independently arbitrary on every chromosome. Answering it needs an outside
    anchor — a relative known to descend from exactly one grandparent.

CAVEAT ON ABSENCE
    "Did not match" is read from the absence of a segment, so it inherits the
    comparison's own threshold. A real 5 cM sharing under a 7 cM threshold
    reads as a non-match and can invert a pattern. Segments near the threshold
    deserve a re-run at a lower one before being trusted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vplab import calibration as C
from vplab import evaluate as V
from vplab import genmap as G
from vplab import intake
from vplab import phase_hmm as H

from tools.phase_real import load_kit, pair_calls


def phase_trio(kits, siblings, emissions, chromosomes, thinning_cm, sex_map):
    """Per-marker partition and posterior for both sides, per chromosome."""
    common = None
    for name in siblings:
        keys = kits[name][["rsid", "chromosome", "position"]]
        common = keys if common is None else common.merge(keys, on=list(keys.columns))
    common = common[common["chromosome"].isin(chromosomes)]
    common = common.drop_duplicates(subset=("rsid", "chromosome", "position"))
    common = common.sort_values(["chromosome", "position"]).reset_index(drop=True)
    if sex_map is not None:
        common = sex_map.annotate(common)

    aligned = {
        name: common.merge(kits[name], on=("rsid", "chromosome", "position"),
                           how="left")[["allele1", "allele2"]]
        for name in siblings
    }
    pairs = [(i, j) for i in range(len(siblings)) for j in range(i + 1, len(siblings))]
    calls = np.stack([V.match_calls(aligned[siblings[i]], aligned[siblings[j]])
                      for i, j in pairs])

    phased = {}
    for chrom in chromosomes:
        rows = np.flatnonzero((common["chromosome"] == chrom).values)
        if len(rows) < 100:
            continue
        keep = rows[H.thin_markers(common["cm"].values[rows], thinning_cm)]
        sex_cm = {}
        if {"cm_male", "cm_female"} <= set(common.columns):
            sex_cm = {"cm_male": common["cm_male"].values[keep],
                      "cm_female": common["cm_female"].values[keep]}
        path, posterior, states = H.phase_chromosome(
            calls[:, keep], pairs, len(siblings), common["cm"].values[keep],
            emissions, **sex_cm)

        entry = {"position": common["position"].values[keep]}
        for side, index in (("paternal", 0), ("maternal", 1)):
            bits = np.array([states[s][index] for s in path])
            # A plain list of tuples, not an object array: numpy would turn the
            # tuples back into arrays and they would stop being hashable.
            called = [tuple((r ^ r[0]).tolist()) for r in bits]
            keys, probs = H.partition_posterior(posterior, states, index)
            lookup = {k: i for i, k in enumerate(keys)}
            entry[side] = called
            entry[side + "_post"] = np.array([
                probs[t, lookup[called[t]]] for t in range(len(called))])
        phased[chrom] = entry
    return phased


def _halves(partition):
    """The two sibling groups a canonical partition describes."""
    partition = np.asarray(partition)
    return frozenset(np.flatnonzero(partition == 0)), frozenset(
        np.flatnonzero(partition == 1))


def merge_overlapping(chrom_rows, index):
    """
    Collapse a chromosome's rows into physical segments.

    One shared segment is reported once per sibling who matches over it, with
    slightly different endpoints each time. Scoring those rows separately would
    count a single piece of evidence three times — and inflate the apparent cM,
    which is how a match sharing ~180 cM came out with "402 cM scored".

    Overlapping rows are therefore merged into one region, and the matching
    pattern is the set of siblings contributing to it.
    """
    ordered = chrom_rows.sort_values("start")
    merged = []
    for row in ordered.itertuples():
        if merged and row.start <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], row.end)
            merged[-1]["members"].add(index[row.sibling])
            merged[-1]["cM"] = max(merged[-1]["cM"], float(row.cM))
        else:
            merged.append({"start": int(row.start), "end": int(row.end),
                           "cM": float(row.cM), "members": {index[row.sibling]}})
    return merged


def votes_for_match(frame, phased, siblings, min_markers=20):
    """One row per physical segment, with the side it discriminates (if any)."""
    index = {name: i for i, name in enumerate(siblings)}
    rows = []
    for chrom, chrom_rows in frame.groupby("chromosome"):
        if chrom not in phased:
            continue
        entry = phased[chrom]
        positions = entry["position"]
        for seg in merge_overlapping(chrom_rows, index):
            sel = np.flatnonzero((positions >= seg["start"]) & (positions <= seg["end"]))
            if len(sel) < min_markers:
                continue
            pattern = frozenset(seg["members"])
            seg = pd.Series(seg)
            verdict, group, confidence = "ambiguous", None, 0.0
            explains = {}
            for side in ("paternal", "maternal"):
                calls = [entry[side][t] for t in sel]
                posts = entry[side + "_post"][sel]
                majority = pd.Series(calls).mode()
                if majority.empty:
                    continue
                majority = majority.iloc[0]
                if pattern in _halves(majority):
                    explains[side] = float(np.mean(posts))
            if len(explains) == 1:
                verdict = "discriminating"
                group, confidence = next(iter(explains.items()))
            elif not explains:
                verdict = "unexplained"
            rows.append({
                "chromosome": int(chrom), "start": int(seg["start"]),
                "end": int(seg["end"]), "cM": float(seg["cM"]),
                "pattern": "".join("1" if i in pattern else "0"
                                   for i in range(len(siblings))),
                "verdict": verdict, "side": group,
                "posterior": round(confidence, 3), "markers": len(sel),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["chromosome", "start"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--kits", default=str(intake.KITS_DIR))
    ap.add_argument("--siblings", nargs="+", required=True)
    ap.add_argument("--unrelated", nargs=2, required=True)
    ap.add_argument("--parent-child", nargs=2, required=True)
    ap.add_argument("--thinning-cm", type=float, default=H.DEFAULT_THINNING_CM)
    ap.add_argument("--calibrator", default="data/calibration_segment.json")
    ap.add_argument("--min-chromosomes", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    segments = pd.read_csv(args.segments)
    chromosomes = sorted(segments["chromosome"].unique())

    sex_map = G.SexSpecificMap.load()
    if sex_map is None:
        raise SystemExit("build the sex-specific map first: python -m tools.build_sex_map")
    calibrator = (C.IsotonicCalibrator.from_json(args.calibrator)
                  if Path(args.calibrator).exists() else None)

    engine = intake._load_engine()
    names = sorted(set(args.siblings) | set(args.unrelated) | set(args.parent_child))
    kits = {name: load_kit(engine, args.kits, name) for name in names}
    emissions = H.calibrate_emissions(
        pair_calls(kits[args.unrelated[0]], kits[args.unrelated[1]]),
        pair_calls(kits[args.parent_child[0]], kits[args.parent_child[1]]))

    phased = phase_trio(kits, args.siblings, emissions, chromosomes,
                        args.thinning_cm, sex_map)

    all_rows = []
    for match, frame in segments.groupby("match"):
        table = votes_for_match(frame, phased, args.siblings)
        if table.empty:
            print(f"\n### {match}: no segment had enough markers to score")
            continue
        table.insert(0, "match", match)
        if calibrator is not None:
            table["partition_conf"] = np.where(
                table.posterior > 0,
                calibrator.transform(table.posterior.values).round(3), 0.0)
        else:
            table["partition_conf"] = table["posterior"]
        all_rows.append(table)

        discriminating = table[table.verdict == "discriminating"]
        print(f"\n### {match}  ({table.cM.sum():.1f} cM over "
              f"{len(table)} physical segments)")
        print(table[["chromosome", "cM", "pattern", "verdict", "side",
                     "partition_conf"]].to_string(index=False))

        tally = discriminating.groupby("side")["chromosome"].nunique()
        unexplained = int((table.verdict == "unexplained").sum())
        print(f"  chromosomes voting: " +
              ", ".join(f"{k} {v}" for k, v in tally.items()) +
              f"   |   unexplained segments: {unexplained}/{len(table)}")

        # The per-vote probability is ASSUMED, not measured: segment-level
        # partition accuracy is known (84.6%) but the accuracy of a *side*
        # attribution has never been measured, and it is a different question
        # -- it needs both sides' partitions to be right, not one. Reported at
        # two assumptions so the reader can see whether the verdict turns on it.
        for label, accuracy in (("optimistic", 0.85), ("conservative", 0.72)):
            call = H.assign_side(
                [(r.chromosome, r.side, accuracy) for r in discriminating.itertuples()],
                min_chromosomes=args.min_chromosomes, n_groups=2)
            print(f"  [{label} {accuracy:.2f}/vote] -> {call}")

    if args.out and all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    print("\nPattern columns are " + "/".join(args.siblings) + ".")
    print("A side is the trio's father's line or their mother's. Which is which,\n"
          "and which grandparent inside a side, both need an outside anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
