# -*- coding: utf-8 -*-
"""
Measure the phasing model against simulated ground truth, and fit its calibrator.

    python -m tools.evaluate_model --families 8 --out-dir data

Every accuracy figure quoted in this repository is produced by this script.
The rule is deliberate: the first set of published figures did not survive
being re-derived, and two of them were wrong in ways nobody could have spotted
by reading the code.

Calibration is scored leave-one-family-out. Fitting a calibrator and reporting
its error on the same families would be a tautology, and the interesting
question is precisely whether a calibration learned on some families transfers
to a family it has never seen — which is what a real user is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vplab import calibration as C
from vplab import evaluate as E
from vplab import genmap as G
from vplab import phase_hmm as H
from vplab import simulate as S

LITERATURE_PER_GAMETE = {"paternal": 26.1, "maternal": 40.9}


def one_family(gmap, sex_map, seed, markers_per_chrom, thinning_cm,
               no_call_rate, error_rate, n_siblings, windows):
    markers = S.build_marker_panel(
        gmap, n_markers_per_chrom=markers_per_chrom,
        rng=np.random.default_rng(100_000 + seed), sex_map=sex_map)
    ped = S.simulate_pedigree(markers, n_siblings=n_siblings, seed=seed,
                              no_call_rate=no_call_rate, error_rate=error_rate)
    emissions = E.calibrate_from_pedigree(ped)
    results, _, siblings = E.phase_pedigree(
        ped, emissions=emissions, thinning_cm=thinning_cm)

    marker = E.marker_accuracy(ped, results, siblings)
    segment = E.segment_accuracy(ped, results, siblings, n_windows=windows, seed=seed)
    crossovers = E.crossover_counts(ped, results, siblings)
    for frame in (marker, segment, crossovers):
        frame["family"] = seed
    return marker, segment, crossovers, emissions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", type=int, default=8)
    ap.add_argument("--siblings", type=int, default=3)
    ap.add_argument("--markers-per-chrom", type=int, default=12000)
    ap.add_argument("--thinning-cm", type=float, default=H.DEFAULT_THINNING_CM)
    ap.add_argument("--no-call-rate", type=float, default=0.002)
    ap.add_argument("--error-rate", type=float, default=0.0002)
    ap.add_argument("--windows", type=int, default=120)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args(argv)

    sex_map = G.SexSpecificMap.load()
    if sex_map is None:
        print("WARNING: no sex-specific map; results will reflect the weaker "
              "sex-averaged model. Build it with `python -m tools.build_sex_map`.")
    gmap = S.load_map("data/min_map.txt", chromosomes=list(range(1, 23)))

    seeds = [11 + 17 * i for i in range(args.families)]
    markers, segments, crossovers = [], [], []
    for seed in seeds:
        m, s, c, em = one_family(
            gmap, sex_map, seed, args.markers_per_chrom, args.thinning_cm,
            args.no_call_rate, args.error_rate, args.siblings, args.windows)
        markers.append(m); segments.append(s); crossovers.append(c)
        print(f"  family {seed}: marker {m.correct.mean():.3f}  "
              f"segment {s.correct.mean():.3f}  error rate {em.error_rate:.5f}")
    marker = pd.concat(markers, ignore_index=True)
    segment = pd.concat(segments, ignore_index=True)
    crossover = pd.concat(crossovers, ignore_index=True)

    n_gametes = args.families * args.siblings
    print("\n=== crossover recovery ===")
    grouped = crossover.groupby("side")[["inferred", "true"]].sum()
    for side, row in grouped.iterrows():
        print(f"  {side:9s} inferred {row.inferred / n_gametes:5.1f} "
              f"true {row.true / n_gametes:5.1f} per gamete   "
              f"recovered {row.inferred / row.true:.2f}   "
              f"literature {LITERATURE_PER_GAMETE[side]:.1f}")

    print("\n=== accuracy ===")
    print(f"  marker  {marker.correct.mean():.1%}  (n={len(marker):,})")
    print(f"  segment {segment.correct.mean():.1%}  (n={len(segment):,}, chance 25%)")
    per_family = segment.groupby("family")["correct"].mean()
    print(f"  segment accuracy across families: {per_family.min():.1%} - "
          f"{per_family.max():.1%} (sd {per_family.std():.3f})")

    print("\n=== calibration, leave-one-family-out ===")
    report = {}
    for label, frame in (("segment", segment), ("marker", marker)):
        raw, cal = [], []
        for held in seeds:
            fit = frame[frame.family != held]
            test = frame[frame.family == held]
            calibrator = C.IsotonicCalibrator.fit(fit.posterior, fit.correct)
            scored = C.evaluate(calibrator, test.posterior.values,
                                test.correct.values.astype(float))
            raw.append(scored["ece_raw"]); cal.append(scored["ece_calibrated"])
        print(f"  {label:8s} ECE raw {np.mean(raw):.3f} -> "
              f"recalibrated {np.mean(cal):.3f}  (held-out mean over "
              f"{len(seeds)} families)")
        report[label] = {"ece_raw": round(float(np.mean(raw)), 4),
                         "ece_calibrated": round(float(np.mean(cal)), 4),
                         "accuracy": round(float(frame.correct.mean()), 4)}
        # Ship a calibrator fitted on everything.
        final = C.IsotonicCalibrator.fit(
            frame.posterior, frame.correct,
            note=f"{label} level, {args.families} simulated families, "
                 f"thinning {args.thinning_cm} cM")
        final.to_json(Path(args.out_dir) / f"calibration_{label}.json")

    # The irreducible part: even a perfect global calibrator cannot remove
    # variation in accuracy *between* families, only the average bias.
    gaps = segment.groupby("family").apply(
        lambda g: g.posterior.mean() - g.correct.mean(), include_groups=False)
    print(f"\n  mean posterior bias {gaps.mean():+.3f} (removable by calibration)")
    print(f"  between-family spread sd {gaps.std():.3f} (not removable without "
          f"run-level information)")

    report["crossovers"] = {
        side: round(float(row.inferred / row.true), 3)
        for side, row in grouped.iterrows()
    }
    report["config"] = vars(args)
    Path(args.out_dir, "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out_dir}/evaluation.json and calibration_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
