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

from . import genmap as _G

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
    relatives: dict = field(default_factory=dict)  # 'Father'/'Mother'/'Unrelated' -> genotype frame

    @property
    def siblings(self):
        return sorted(self.genotypes)


def load_map(map_path, chromosomes=None):
    """Load min_map.txt-format genetic map: Chromosome / Position / cM."""
    raw = pd.read_csv(map_path, sep="\t", header=0)
    if chromosomes is not None:
        raw = raw[raw["Chromosome"].isin(chromosomes)]
    return raw.sort_values(["Chromosome", "Position"]).reset_index(drop=True)


def build_marker_panel(genetic_map, n_markers_per_chrom=6000, rng=None,
                       sex_map=None):
    """
    Lay a realistic SNP panel over the mapped region of each chromosome.

    Consumer chips carry roughly 600-700k markers genome-wide. Spacing is made
    uneven on purpose: uniform spacing hides exactly the failure modes (sparse
    regions, map edges) that matter.

    Pass `sex_map` (a genmap.SexSpecificMap) to attach `cm_male` and
    `cm_female` as well; without it the panel carries only the sex-averaged
    coordinate and meiosis falls back to global rate scales.
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
    panel = pd.concat(frames, ignore_index=True)
    if sex_map is not None:
        panel = sex_map.annotate(panel)
    return panel


# Minor-allele-frequency spectrum of the marker panel, as Beta(a, b).
#
# This is not a cosmetic detail: it sets how often two *unrelated* people
# coincide by chance, which is the noise floor the whole method works against.
# The default is calibrated against a real unrelated pair in the reference
# family (FIR at 74.0% of markers, NIR at 3.3%); Beta(0.25, 1.2) reproduces
# 73.9% / 3.6%. Consumer chips are dominated by low-frequency markers, so a
# flat MAF distribution makes simulated data far more discriminating than any
# real dataset.
MAF_BETA = (0.25, 1.2)
MAF_FLOOR, MAF_CEILING = 0.002, 0.5


def assign_allele_frequencies(n_markers, rng, maf_beta=MAF_BETA):
    """
    Reference/alternate alleles and a minor-allele frequency for each marker.

    Allele identity is a property of the *marker*, not of the individual: every
    person genotyped on a given chip is scored against the same two alleles at
    a locus. Drawing a fresh reference/alternate pair per founder — as an
    earlier version of this file did — gives each founder a private biallelic
    system, so two unrelated people almost never share an allele. Measured, it
    put an unrelated pair at 56% no-match where real kits sit at 3.3%, handing
    the downstream model ~17x more exclusion evidence than reality supplies.
    Exclusions are the strongest signal visual phasing has, so that error makes
    simulated data *easier* than real data, not noisier.
    """
    ref_idx = rng.integers(0, 4, size=n_markers)
    # Alternate allele differs from reference.
    alt_idx = (ref_idx + rng.integers(1, 4, size=n_markers)) % 4
    maf = np.clip(rng.beta(*maf_beta, size=n_markers), MAF_FLOOR, MAF_CEILING)
    return ref_idx, alt_idx, maf


def _draw_founder_haplotypes(panel, rng):
    """
    Two independent haplotypes for one founder, drawn from the panel's own
    allele frequencies. Founders are unrelated to each other, so their
    haplotypes are independent draws — but from a *shared* allele model.
    """
    ref_idx, alt_idx, maf = panel
    n_markers = len(maf)
    hap0 = np.where(rng.random(n_markers) < maf, alt_idx, ref_idx)
    hap1 = np.where(rng.random(n_markers) < maf, alt_idx, ref_idx)
    return hap0, hap1


def meiosis_cm(markers, sex):
    """
    The genetic coordinate a meiosis of this sex actually runs on.

    Prefers the sex-specific column when the panel carries one. Falling back to
    a scaled sex-averaged map gets the *number* of crossovers right and their
    *placement* wrong, and placement is what distinguishes a paternal
    chromosome from a maternal one.
    """
    column = "cm_female" if sex == "female" else "cm_male"
    if column in markers.columns and markers[column].notna().all():
        return markers[column].values, 1.0
    male_scale, female_scale = _G.fallback_scales()
    return markers["cm"].values, (female_scale if sex == "female" else male_scale)


# Shape parameter of the gamma renewal model of crossover placement. nu = 1 is
# a Poisson process (no interference); human estimates cluster around 4-6.
INTERFERENCE_NU = 4.0


def _gamma_renewal_crossovers(start_cm, end_cm, rng, nu=INTERFERENCE_NU,
                              burn_in_cm=300.0):
    """
    Crossover positions along one chromosome, with interference.

    Crossovers are not independent: one suppresses others nearby, so tight
    doubles are far rarer than a Poisson process predicts. The standard model
    is a gamma renewal process — inter-crossover distances are Gamma(nu,
    1/(nu*lambda)) rather than exponential, which keeps the mean rate at the
    map's own (one crossover per Morgan) while pushing the *variance* down.

    This replaces an earlier scheme that drew Poisson crossovers and then
    deleted any falling within 10 cM of the previous one. Deleting events does
    suppress tight doubles, but it also suppresses the rate: measured, it
    produced 37.4 maternal crossovers per gamete where the map specifies 41.
    Simulated truth was quietly wrong, so every count scored against it was
    scored against the wrong target.

    A burn-in before the chromosome start avoids the edge bias of beginning the
    renewal process exactly at position zero.
    """
    scale_cm = 100.0 / nu          # mean spacing stays 100 cM = 1 Morgan
    position = start_cm - burn_in_cm
    out = []
    while True:
        position += rng.gamma(nu, scale_cm)
        if position > end_cm:
            return np.array(out)
        if position >= start_cm:
            out.append(position)


def _simulate_meiosis(markers, rng, sex="female", interference=True,
                      interference_nu=INTERFERENCE_NU):
    """
    Produce one gamete: a mosaic of the parent's two haplotypes.

    Crossovers are drawn as a Poisson process along that sex's genetic map,
    which is what makes the same map the right transition model for an HMM
    downstream.

    Returns (selector, crossover_positions) where selector[i] in {0,1} says
    which parental haplotype marker i came from.
    """
    cm_all, scale = meiosis_cm(markers, sex)

    selector = np.zeros(len(markers), dtype=np.int8)
    crossovers = {}

    for chrom, group in markers.groupby("chromosome"):
        idx = group.index.values
        cm = cm_all[idx]
        length_cm = cm[-1] - cm[0]
        if length_cm <= 0:
            continue

        # The scale factor is 1.0 whenever a true sex-specific map is in use;
        # it only compensates for a sex-averaged map (see meiosis_cm).
        if interference:
            positions_cm = _gamma_renewal_crossovers(
                cm[0], cm[0] + length_cm * scale, rng, interference_nu)
            positions_cm = cm[0] + (positions_cm - cm[0]) / scale
        else:
            n_co = rng.poisson((length_cm / 100.0) * scale)
            positions_cm = np.sort(rng.uniform(cm[0], cm[-1], size=n_co))

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


def _to_genotype_frame(hap0, hap1, rng, no_call_rate=0.0, error_rate=0.0):
    """
    Two haplotypes -> a vendor-style genotype frame, with artefacts injected.

    A miscall replaces one allele with a random base; a no-call blanks the
    whole genotype, which is how vendors actually emit missing data.
    """
    geno = pd.DataFrame({"allele1": BASES[hap0], "allele2": BASES[hap1]})
    n = len(geno)
    if error_rate > 0:
        for col in ("allele1", "allele2"):
            hit = rng.random(n) < (error_rate / 2)
            geno.loc[hit, col] = BASES[rng.integers(0, 4, size=hit.sum())]
    if no_call_rate > 0:
        hit = rng.random(n) < no_call_rate
        geno.loc[hit, "allele1"] = "-"
        geno.loc[hit, "allele2"] = "-"
    return geno


def simulate_pedigree(
    markers,
    n_siblings=3,
    seed=0,
    no_call_rate=0.0,
    error_rate=0.0,
    maf_beta=MAF_BETA,
):
    """
    Simulate 4 grandparents -> 2 parents -> n_siblings, with ground truth.

    no_call_rate and error_rate inject the two artefacts that matter for the
    correctness of a segment caller: missing genotypes, and miscalled ones.

    Also emits both parents and one unrelated individual under `relatives`.
    Those are not decoration: a parent-child pair is IBD1 at every locus and an
    unrelated pair is IBD0 everywhere, which is exactly what calibrates the
    emission model. Without them, evaluating the HMM on simulated data would
    have to assume the emission table it is meant to be testing.
    """
    rng = np.random.default_rng(seed)
    n = len(markers)

    # --- The marker panel's own allele model ----------------------------
    panel = assign_allele_frequencies(n, rng, maf_beta)

    # --- Founders: the four grandparents -------------------------------
    founders = {}
    for gp in GRANDPARENTS:
        founders[gp] = _draw_founder_haplotypes(panel, rng)

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
        genotypes[name] = _to_genotype_frame(a1, a2, rng, no_call_rate, error_rate)
        origins[name] = pd.DataFrame(
            {"paternal": origin_pair[0], "maternal": origin_pair[1]}
        )
        crossovers[name] = co_pair

    # --- Calibration relatives -------------------------------------------
    relatives = {
        "Father": _to_genotype_frame(*parents["father"]["haps"], rng,
                                     no_call_rate, error_rate),
        "Mother": _to_genotype_frame(*parents["mother"]["haps"], rng,
                                     no_call_rate, error_rate),
        "Unrelated": _to_genotype_frame(*_draw_founder_haplotypes(panel, rng), rng,
                                        no_call_rate, error_rate),
    }

    return Pedigree(
        markers=markers,
        genotypes=genotypes,
        origins=origins,
        crossovers=crossovers,
        founders=founders,
        relatives=relatives,
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
