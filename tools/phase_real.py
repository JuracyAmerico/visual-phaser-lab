# -*- coding: utf-8 -*-
"""
Run the grandparental-assignment HMM on real kits.

    python -m tools.phase_real \
        --kits ~/DNA-Kits --siblings A B C \
        --unrelated A M --parent-child A CHILD \
        --out segments.csv

No names, paths or family structure are baked in — everything comes from the
command line, so this file is safe to publish and the same command reproduces
the numbers quoted anywhere in this repo.

The two calibration pairs are what make the output honest rather than assumed:

    --unrelated     two people with no shared ancestry. IBD0 at every locus, so
                    their observed match distribution *is* the IBD0 emission row.
    --parent-child  a parent and their child. IBD1 at every locus by definition,
                    which gives the IBD1 row — and any apparent exclusion in that
                    pair cannot be real, so it measures the genotyping error rate.

Reported crossover counts are the diagnostic to read first. Human meiosis is
tightly constrained (~26 crossovers per male gamete, ~41 per female), so a
count far off that says the transition model is wrong before any segment call
is worth reading.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vplab import calibration as C
from vplab import genmap as G
from vplab import intake
from vplab import phase_hmm as H

# Literature genome-wide autosomal crossover counts per gamete.
EXPECTED_PER_GAMETE = {"paternal": 26.1, "maternal": 40.9}


def load_kit(engine, kits_dir, name):
    """Find `<name>_raw_dna*` anywhere under the kit directory and load it."""
    matches = sorted(Path(kits_dir).glob(f"**/{name}_raw_dna*"))
    if not matches:
        raise FileNotFoundError(f"no kit named {name!r} under {kits_dir}")
    path = matches[0]
    _, frame = engine.agnostic_load_individual_dna(
        path.name.split("_raw_dna")[0], str(path.parent), "X")
    # Consumer exports repeat a few hundred rsIDs at the same coordinate. Left
    # unhandled, merging two kits becomes a partial cartesian product and the
    # per-kit marker arrays come out different lengths, so the comparison
    # cannot be aligned at all. Measured, deduplication changes only the
    # alignment: the parent-child exclusion rate is 1.022% before and after.
    return frame.drop_duplicates(subset=("rsid", "chromosome", "position"), keep="first")


def pair_calls(frame_a, frame_b):
    """Per-marker match class on the markers the two kits have in common."""
    merged = frame_a.merge(frame_b, on=("rsid", "chromosome", "position"),
                           suffixes=("_1", "_2"))
    geno_a = merged[["allele1_1", "allele2_1"]].rename(
        columns={"allele1_1": "allele1", "allele2_1": "allele2"})
    geno_b = merged[["allele1_2", "allele2_2"]].rename(
        columns={"allele1_2": "allele1", "allele2_2": "allele2"})
    from vplab.evaluate import match_calls
    return match_calls(geno_a, geno_b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kits", default=str(intake.KITS_DIR))
    ap.add_argument("--siblings", nargs="+", required=True)
    ap.add_argument("--unrelated", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--parent-child", nargs=2, required=True, metavar=("PARENT", "CHILD"))
    ap.add_argument("--chromosomes", default="1-22")
    ap.add_argument("--thinning-cm", type=float, default=H.DEFAULT_THINNING_CM)
    ap.add_argument("--map-function", default="kosambi", choices=sorted(H.MAP_FUNCTIONS))
    ap.add_argument("--calibrator", default="data/calibration_segment.json")
    ap.add_argument("--min-cm", type=float, default=7.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    lo, _, hi = args.chromosomes.partition("-")
    chromosomes = list(range(int(lo), int(hi or lo) + 1))

    sex_map = G.SexSpecificMap.load()
    if sex_map is None:
        print("WARNING: no sex-specific map; falling back to global rate scales.\n"
              "         Side labels are then only weakly identifiable — build it\n"
              "         with `python -m tools.build_sex_map`.", file=sys.stderr)

    calibrator = None
    if Path(args.calibrator).exists():
        calibrator = C.IsotonicCalibrator.from_json(args.calibrator)

    engine = intake._load_engine()
    names = sorted(set(args.siblings) | set(args.unrelated) | set(args.parent_child))
    kits = {name: load_kit(engine, args.kits, name) for name in names}
    for name in names:
        print(f"  loaded {name}: {len(kits[name]):,} markers", file=sys.stderr)

    emissions = H.calibrate_emissions(
        pair_calls(kits[args.unrelated[0]], kits[args.unrelated[1]]),
        pair_calls(kits[args.parent_child[0]], kits[args.parent_child[1]]),
    )
    print(f"\nemissions: error rate {emissions.error_rate:.5f}\n  {emissions.source}",
          file=sys.stderr)
    print("  P(obs | IBD) rows IBD0/1/2, cols FIR/HIR/NIR:", file=sys.stderr)
    print("  " + np.array2string(np.round(emissions.table, 4)).replace("\n", "\n  "),
          file=sys.stderr)

    # Markers common to every sibling, in genomic order.
    common = None
    for name in args.siblings:
        keys = kits[name][["rsid", "chromosome", "position"]]
        common = keys if common is None else common.merge(keys, on=list(keys.columns))
    common = common[common["chromosome"].isin(chromosomes)]
    common = common.drop_duplicates(subset=("rsid", "chromosome", "position"))
    common = common.sort_values(["chromosome", "position"]).reset_index(drop=True)
    if sex_map is not None:
        common = sex_map.annotate(common)
    print(f"\n{len(common):,} markers shared by all {len(args.siblings)} siblings",
          file=sys.stderr)

    aligned = {}
    for name in args.siblings:
        merged = common.merge(kits[name], on=("rsid", "chromosome", "position"), how="left")
        aligned[name] = merged[["allele1", "allele2"]]

    from vplab.evaluate import match_calls
    pairs = [(i, j) for i in range(len(args.siblings))
             for j in range(i + 1, len(args.siblings))]
    calls = np.stack([
        match_calls(aligned[args.siblings[i]], aligned[args.siblings[j]]) for i, j in pairs
    ])

    segments, crossovers = [], []
    for chrom in chromosomes:
        rows = np.flatnonzero((common["chromosome"] == chrom).values)
        if len(rows) < 100:
            continue
        cm_all = common["cm"].values[rows]
        keep = rows[H.thin_markers(cm_all, args.thinning_cm)]
        sex_cm = {}
        if {"cm_male", "cm_female"} <= set(common.columns):
            sex_cm = {"cm_male": common["cm_male"].values[keep],
                      "cm_female": common["cm_female"].values[keep]}

        path, posterior, states = H.phase_chromosome(
            calls[:, keep], pairs, len(args.siblings), common["cm"].values[keep],
            emissions, map_function=args.map_function, **sex_cm)

        segments.append(H.assign_grandparents(
            path, states, common["position"].values[keep], common["cm"].values[keep],
            len(args.siblings), min_cm=args.min_cm, posterior=posterior,
            calibrator=calibrator, chromosome=chrom))

        for side, idx in (("paternal", 0), ("maternal", 1)):
            bits = np.array([states[s][idx] for s in path])
            for k, name in enumerate(args.siblings):
                crossovers.append({"chromosome": chrom, "side": side, "sibling": name,
                                   "crossovers": int(np.count_nonzero(np.diff(bits[:, k])))})
        print(f"  chr{chrom:<2d} {len(keep):>5,} markers thinned from {len(rows):>6,}",
              file=sys.stderr)

    segments = pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()
    crossovers = pd.DataFrame(crossovers)

    print("\n=== crossovers per gamete (side labels are a partition, not names) ===")
    summary = crossovers.groupby(["side", "sibling"])["crossovers"].sum().unstack()
    print(summary.to_string())
    per_side = crossovers.groupby("side")["crossovers"].sum() / len(args.siblings)
    for side, observed in per_side.items():
        expected = EXPECTED_PER_GAMETE[side]
        print(f"  {side:9s} {observed:5.1f} per gamete   literature {expected:.1f}"
              f"   ratio {observed / expected:.2f}")
    print("\n  Which group is truly paternal is NOT established by these counts —\n"
          "  the model is told one side recombines less. It takes an outside\n"
          "  anchor: a relative descending from exactly one grandparent.")

    if not segments.empty:
        print(f"\n=== segments >= {args.min_cm} cM ===")
        print(f"  {len(segments):,} segments, "
              f"{segments['cM'].sum():,.0f} cM total")
        if "confidence" in segments:
            for lo_, hi_ in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
                sel = segments[(segments.confidence >= lo_) & (segments.confidence < hi_)]
                if len(sel):
                    print(f"  confidence {lo_:.2f}-{hi_:.2f}: {len(sel):4,} segments, "
                          f"{sel['cM'].sum():7,.0f} cM")
    if args.out:
        segments.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
