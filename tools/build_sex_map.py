# -*- coding: utf-8 -*-
"""
Build a condensed sex-specific genetic map.

Downloads the refined European maps of

    Bhérer, C., Campbell, C. L. & Auton, A. Refined genetic maps reveal sexual
    dimorphism in human meiotic recombination at multiple scales.
    Nat. Commun. 8, 14994 (2017).  doi:10.1038/ncomms14994
    https://github.com/cbherer/Bherer_etal_SexualDimorphismRecombination

and condenses them into one build-37 table with male, female and sex-averaged
positions per marker.

The upstream tarball is 37 MB across 781k rows and carries no stated
redistribution licence, so it is *not* vendored here — this script fetches it
and writes a derived file that `.gitignore` keeps out of the repo. Attribution
above is the citation to use for any figure produced from it.

Why sex-specific maps matter here, measured rather than assumed: with a
sex-averaged map plus a global rate scale, the two parental sides differ only
in how *often* they recombine. That is weak evidence, and phasing each
chromosome independently made the side labels flip on 2 of 10 simulated
chromosomes — silently attributing a chromosome's paternal segments to the
maternal grandparents. Male and female maps differ in *shape*, not just total
length (male recombination is strongly telomeric), which is per-chromosome
positional evidence of which side is which.

    python -m tools.build_sex_map [--out data/sex_specific_map.txt]
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/cbherer/"
    "Bherer_etal_SexualDimorphismRecombination/master/"
    "Refined_EUR_genetic_map_b37.tar.gz"
)
AUTOSOMES = range(1, 23)


def fetch(url=URL, cache=None):
    """Download the tarball, caching it so re-runs are free."""
    if cache is not None and Path(cache).exists():
        return Path(cache).read_bytes()
    with urllib.request.urlopen(url, timeout=600) as handle:
        blob = handle.read()
    if cache is not None:
        Path(cache).write_bytes(blob)
    return blob


def _read_member(tar, sex, chrom):
    name = f"Refined_EUR_genetic_map_b37/{sex}_chr{chrom}.txt"
    with tar.extractfile(name) as handle:
        frame = pd.read_csv(handle, sep="\t")
    return frame[["pos", "cM"]].rename(columns={"pos": "position", "cM": sex})


def condense(positions, curves, tolerance_cm=0.01):
    """
    Drop points a linear interpolation already reproduces.

    Douglas-Peucker on the cM curve rather than fixed decimation: the map is
    flat across cold regions and steep at hotspots, so uniform thinning would
    spend rows where nothing happens and lose resolution exactly where the
    transition model needs it. Every retained point is one whose removal would
    move some interpolated position by more than `tolerance_cm`.
    """
    keep = np.zeros(len(positions), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(positions) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        span = positions[hi] - positions[lo]
        if span <= 0:
            continue
        frac = (positions[lo + 1:hi] - positions[lo]) / span
        worst_i, worst_d = None, 0.0
        for curve in curves:
            approx = curve[lo] + frac * (curve[hi] - curve[lo])
            dev = np.abs(curve[lo + 1:hi] - approx)
            i = int(np.argmax(dev))
            if dev[i] > worst_d:
                worst_d, worst_i = float(dev[i]), lo + 1 + i
        if worst_i is not None and worst_d > tolerance_cm:
            keep[worst_i] = True
            stack.append((lo, worst_i))
            stack.append((worst_i, hi))
    return keep


def build(blob, tolerance_cm=0.01):
    frames = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for chrom in AUTOSOMES:
            merged = None
            for sex in ("male", "female", "sexavg"):
                part = _read_member(tar, sex, chrom)
                merged = part if merged is None else merged.merge(part, on="position")
            merged = merged.sort_values("position").reset_index(drop=True)
            pos = merged["position"].values.astype(float)
            curves = [merged[s].values.astype(float) for s in ("male", "female", "sexavg")]
            keep = condense(pos, curves, tolerance_cm)
            out = merged.loc[keep].copy()
            out.insert(0, "chromosome", chrom)
            frames.append(out)
            print(f"  chr{chrom:<2d} {len(merged):>7,} -> {len(out):>6,} rows   "
                  f"male {curves[0][-1]:7.1f} cM  female {curves[1][-1]:7.1f} cM",
                  file=sys.stderr)
    table = pd.concat(frames, ignore_index=True)
    return table.rename(columns={
        "male": "male_cm", "female": "female_cm", "sexavg": "sexavg_cm"
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sex_specific_map.txt")
    ap.add_argument("--cache", default=None, help="path to keep the downloaded tarball")
    ap.add_argument("--tolerance-cm", type=float, default=0.01)
    args = ap.parse_args(argv)

    print(f"fetching {URL}", file=sys.stderr)
    table = build(fetch(cache=args.cache), args.tolerance_cm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False, float_format="%.6f")

    totals = table.groupby("chromosome")[["male_cm", "female_cm", "sexavg_cm"]].max().sum()
    print(f"\nwrote {out}  ({len(table):,} rows, {out.stat().st_size / 1e6:.1f} MB)",
          file=sys.stderr)
    print(f"genome-wide: male {totals.male_cm:.1f} cM, female {totals.female_cm:.1f} cM, "
          f"sex-averaged {totals.sexavg_cm:.1f} cM", file=sys.stderr)
    print(f"expected ~26 and ~42 crossovers per gamete: "
          f"{totals.male_cm / 100:.1f} / {totals.female_cm / 100:.1f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
