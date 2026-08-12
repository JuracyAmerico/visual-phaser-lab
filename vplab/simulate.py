# -*- coding: utf-8 -*-
"""
Meiosis simulator — ground truth for visual phasing.

Simulates a three-generation pedigree (4 grandparents -> 2 parents -> N
siblings) using a real genetic map, then writes vendor-format raw DNA files
alongside the *known* crossover positions and grandparental assignments.

Why this exists: every visual-phasing tool in the wild asserts its accuracy
rather than measuring it, because real families do not come with known
crossover points. Simulated ones do. That turns "this works well" into
"recovers 97% of crossovers within 0.5 cM", and it makes the correctness of a
segment caller testable instead of arguable.

Nothing here touches real genotypes. Founder haplotypes are drawn from allele
frequencies, so output files are synthetic and safe to commit or share.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASES = np.array(["A", "C", "G", "T"])

# The four grandparents, in the order used throughout: paternal grandfather,
# paternal grandmother, maternal grandfather, maternal grandmother.
GRANDPARENTS = ("PGF", "PGM", "MGF", "MGM")


@dataclass
class Pedigree:
    """A simulated family plus the ground truth that generated it."""

    markers: pd.DataFrame           # rsid, chromosome, position, cm
    genotypes: dict                 # name -> DataFrame(allele1, allele2)
    origins: dict                   # name -> DataFrame(paternal, maternal) of grandparent labels
    crossovers: dict                # name -> {"paternal": {chrom: [pos]}, "maternal": {...}}
    founders: dict = field(default_factory=dict)   # grandparent -> (hap0, hap1)

    @property
    def siblings(self):
        return sorted(self.genotypes)


def load_map(map_path, chromosomes=None):
    """Load min_map.txt-format genetic map: Chromosome / Position / cM."""
    raw = pd.read_csv(map_path, sep="\t", header=0)
    if chromosomes is not None:
        raw = raw[raw["Chromosome"].isin(chromosomes)]
    return raw.sort_values(["Chromosome", "Position"]).reset_index(drop=True)


def build_marker_panel(genetic_map, n_markers_per_chrom=6000, rng=None):
    """
    Lay a realistic SNP panel over the mapped region of each chromosome.

    Consumer chips carry roughly 600-700k markers genome-wide. Spacing is made
    uneven on purpose: uniform spacing hides exactly the failure modes (sparse
    regions, map edges) that matter.
    """
    rng = rng or np.random.default_rng(0)
    frames = []
    for chrom, group in genetic_map.groupby("Chromosome"):
        lo, hi = group["Position"].min(), group["Position"].max()
        # Jittered positions, deduplicated and sorted.
        pos = np.sort(rng.integers(lo, hi, size=n_markers_per_chrom))
        pos = np.unique(pos)
        cm = np.interp(pos, group["Position"].values, group["cM"].values)
        frames.append(
            pd.DataFrame(
                {
                    "rsid": [f"rs{chrom}_{i}" for i in range(len(pos))],
                    "chromosome": chrom,
                    "position": pos,
                    "cm": cm,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _draw_founder_haplotypes(n_markers, rng, maf_range=(0.05, 0.5)):
    """
    Two haplotypes for one founder, drawn from per-marker allele frequencies.

    Markers are biallelic with a random reference/alternate pair, which keeps
    the simulation honest: a classifier that mishandles multi-allelic or
    strand-flipped data will not be flattered by this input.
    """
    ref_idx = rng.integers(0, 4, size=n_markers)
    # Alternate allele differs from reference.
    alt_idx = (ref_idx + rng.integers(1, 4, size=n_markers)) % 4
    maf = rng.uniform(*maf_range, size=n_markers)

    hap0 = np.where(rng.random(n_markers) < maf, alt_idx, ref_idx)
    hap1 = np.where(rng.random(n_markers) < maf, alt_idx, ref_idx)
    return hap0, hap1, ref_idx, alt_idx


def _simulate_meiosis(markers, rng, sex="female", interference=True):
    """
    Produce one gamete: a mosaic of the parent's two haplotypes.

    Crossovers are drawn as a Poisson process along the genetic map, which is
    what makes the map the right transition model for an HMM downstream. Female
    meiosis recombines noticeably more than male; the ratio here (~1.6x) is the
    well-established genome-wide average, applied as a scale factor since this
    map is sex-averaged.

    Returns (selector, crossover_positions) where selector[i] in {0,1} says
    which parental haplotype marker i came from.
    """
    scale = 1.25 if sex == "female" else 0.78

    selector = np.zeros(len(markers), dtype=np.int8)
    crossovers = {}

    for chrom, group in markers.groupby("chromosome"):
        idx = group.index.values
        cm = group["cm"].values
        length_cm = cm[-1] - cm[0]
        if length_cm <= 0:
            continue

        expected = (length_cm / 100.0) * scale
        n_co = rng.poisson(expected)

        if interference and n_co > 1:
            # Crossover interference suppresses closely spaced events. Sampling
            # then enforcing a minimum spacing is a crude but standard stand-in
            # for a full gamma-model; without it, simulated data has far more
            # tight double-crossovers than real meiosis.
            positions_cm = np.sort(rng.uniform(cm[0], cm[-1], size=n_co))
            keep = [positions_cm[0]]
            for p in positions_cm[1:]:
                if p - keep[-1] >= 10.0:   # ~10 cM minimum spacing
                    keep.append(p)
            positions_cm = np.array(keep)
        elif n_co > 0:
            positions_cm = np.sort(rng.uniform(cm[0], cm[-1], size=n_co))
        else:
            positions_cm = np.array([])

        start = rng.integers(0, 2)
        sel = np.full(len(idx), start, dtype=np.int8)
        for p in positions_cm:
            sel[cm >= p] ^= 1
        selector[idx] = sel

        # Record crossovers in physical coordinates for scoring later.
        crossovers[int(chrom)] = [
            int(np.interp(p, cm, group["position"].values)) for p in positions_cm
        ]

    return selector, crossovers


def simulate_pedigree(
    markers,
    n_siblings=3,
    seed=0,
    no_call_rate=0.0,
    error_rate=0.0,
):
    """
    Simulate 4 grandparents -> 2 parents -> n_siblings, with ground truth.

    no_call_rate and error_rate inject the two artefacts that matter for the
    correctness of a segment caller: missing genotypes, and miscalled ones.
    """
    rng = np.random.default_rng(seed)
    n = len(markers)

    # --- Founders: the four grandparents -------------------------------
    founders = {}
    for gp in GRANDPARENTS:
        hap0, hap1, _, _ = _draw_founder_haplotypes(n, rng)
        founders[gp] = (hap0, hap1)

    # --- Parents: each a recombinant of two grandparents ----------------
    # The father's transmitted gamete carries PGF/PGM origin labels; the
    # mother's carries MGF/MGM. Tracking origin per marker is what lets us
    # score a phasing tool against truth.
    parents = {}
    for parent, (gp_a, gp_b), sex in (
        ("father", ("PGF", "PGM"), "male"),
        ("mother", ("MGF", "MGM"), "female"),
    ):
        # Parent's own two haplotypes, one inherited from each of their parents.
        sel_a, _ = _simulate_meiosis(markers, rng, sex="male")
        sel_b, _ = _simulate_meiosis(markers, rng, sex="female")
        hap_from_a = np.where(sel_a == 0, founders[gp_a][0], founders[gp_a][1])
        hap_from_b = np.where(sel_b == 0, founders[gp_b][0], founders[gp_b][1])
        parents[parent] = {
            "haps": (hap_from_a, hap_from_b),
            "origin_labels": (gp_a, gp_b),
            "sex": sex,
        }

    # --- Siblings --------------------------------------------------------
    genotypes, origins, crossovers = {}, {}, {}
    for s in range(n_siblings):
        name = f"Sib{s + 1}"
        allele_pair, origin_pair, co_pair = [], [], {}

        for parent, side in (("father", "paternal"), ("mother", "maternal")):
            info = parents[parent]
            sel, co = _simulate_meiosis(markers, rng, sex=info["sex"])
            transmitted = np.where(sel == 0, info["haps"][0], info["haps"][1])
            labels = np.where(sel == 0, info["origin_labels"][0], info["origin_labels"][1])
            allele_pair.append(transmitted)
            origin_pair.append(labels)
            co_pair[side] = co

        a1, a2 = allele_pair
        geno = pd.DataFrame({"allele1": BASES[a1], "allele2": BASES[a2]})

        # --- Artefact injection ------------------------------------------
        if error_rate > 0:
            # A miscall replaces one allele with a random base. This is the
            # artefact that isolated-mismatch smoothing is meant to absorb.
            for col in ("allele1", "allele2"):
                hit = rng.random(n) < (error_rate / 2)
                geno.loc[hit, col] = BASES[rng.integers(0, 4, size=hit.sum())]

        if no_call_rate > 0:
            # Vendors emit a no-call for the whole genotype, not one allele.
            hit = rng.random(n) < no_call_rate
            geno.loc[hit, "allele1"] = "-"
            geno.loc[hit, "allele2"] = "-"

        genotypes[name] = geno
        origins[name] = pd.DataFrame(
            {"paternal": origin_pair[0], "maternal": origin_pair[1]}
        )
        crossovers[name] = co_pair

    return Pedigree(
        markers=markers,
        genotypes=genotypes,
        origins=origins,
        crossovers=crossovers,
        founders=founders,
    )


def write_raw_dna(pedigree, out_dir, fmt="genera", gzip_output=False):
    """
    Write vendor-format raw DNA files named for Visual Phaser's convention
    (`<Name>_raw_dna...`).

    'genera' emits RSID,CHROMOSOME,POSITION,RESULT — the layout of the real
    Brazilian-vendor export this was built against. '23andme' emits the
    tab-delimited four-column form.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markers = pedigree.markers
    written = []

    for name, geno in pedigree.genotypes.items():
        if fmt == "genera":
            frame = pd.DataFrame(
                {
                    "RSID": markers["rsid"],
                    "CHROMOSOME": markers["chromosome"],
                    "POSITION": markers["position"],
                    "RESULT": geno["allele1"].values + geno["allele2"].values,
                }
            )
            path = out_dir / f"{name}_raw_dna.csv"
            sep = ","
        elif fmt == "23andme":
            frame = pd.DataFrame(
                {
                    "rsid": markers["rsid"],
                    "chromosome": markers["chromosome"],
                    "position": markers["position"],
                    "genotype": geno["allele1"].values + geno["allele2"].values,
                }
            )
            path = out_dir / f"{name}_raw_dna.txt"
            sep = "\t"
        else:
            raise ValueError(f"unknown format: {fmt}")

        if gzip_output:
            path = path.with_suffix(path.suffix + ".gz")
            with gzip.open(path, "wt") as handle:
                frame.to_csv(handle, sep=sep, index=False)
        else:
            frame.to_csv(path, sep=sep, index=False)
        written.append(path)

    return written


def truth_table(pedigree):
    """
    Flatten ground truth to one row per (sibling, chromosome, side, crossover).
    This is what a phasing tool's output gets scored against.
    """
    rows = []
    for name, sides in pedigree.crossovers.items():
        for side, per_chrom in sides.items():
            for chrom, positions in per_chrom.items():
                for pos in positions:
                    rows.append(
                        {"sibling": name, "side": side, "chromosome": chrom, "position": pos}
                    )
    return pd.DataFrame(rows).sort_values(["sibling", "chromosome", "side", "position"])
