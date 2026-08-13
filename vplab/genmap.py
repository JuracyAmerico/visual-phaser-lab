# -*- coding: utf-8 -*-
"""
Sex-specific genetic map.

Male and female meiosis do not merely recombine at different *rates* — they
recombine in different *places*. Male crossovers concentrate near the
telomeres and are suppressed across the middle of a chromosome; female
crossovers are spread far more evenly. Genome-wide the difference is a factor
of 1.57 (2608 cM male vs 4094 cM female), but locally the ratio swings from
below 0.3 to above 3.

That distinction is why this module exists. Using a sex-averaged map plus a
global rate scale — which is what this project did first — leaves the two
parental sides differing only in how often they switch. Measured consequence:
phasing each chromosome independently, the paternal and maternal labels came
out *transposed* on 2 of 10 simulated chromosomes, quietly attributing a whole
chromosome's paternal segments to the maternal grandparents. Map shape is
per-chromosome positional evidence of which side is which, where a rate scale
is only a weak global prior.

The map itself is not vendored; `tools/build_sex_map.py` fetches and condenses
it (Bhérer et al. 2017, Nat. Commun. 8:14994). Everything here degrades
gracefully when it is absent — `load()` returns None and callers fall back to
the sex-averaged map with global scales, which is the older, weaker model.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(os.environ.get(
    "VPLAB_SEX_MAP",
    Path(__file__).resolve().parent.parent / "data" / "sex_specific_map.txt",
))

# Genome-wide autosomal totals of the Bhérer refined European maps, used to
# sanity-check a loaded file and to derive the fallback scales.
MALE_TOTAL_CM, FEMALE_TOTAL_CM, SEXAVG_TOTAL_CM = 2607.6, 4094.3, 3350.9


class SexSpecificMap:
    """Position -> (male cM, female cM, sex-averaged cM), per chromosome."""

    def __init__(self, table):
        self.table = table
        self._by_chrom = {
            int(c): g.sort_values("position")
            for c, g in table.groupby("chromosome")
        }

    @classmethod
    def load(cls, path=None, required=False):
        """
        Load the condensed map, or return None if it has not been built.

        Absence is a normal state, not an error: the map is a 5.7 MB derived
        file kept out of version control. Callers are expected to fall back and
        say so, rather than silently producing worse answers.
        """
        path = Path(path or DEFAULT_PATH)
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"{path} not found — build it with "
                    f"`python -m tools.build_sex_map`"
                )
            return None
        table = pd.read_csv(path, sep="\t")
        missing = {"chromosome", "position", "male_cm", "female_cm", "sexavg_cm"} - set(table.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return cls(table)

    @property
    def chromosomes(self):
        return sorted(self._by_chrom)

    def interpolate(self, chromosome, positions):
        """
        Genetic positions for physical positions on one chromosome.

        Linear interpolation between mapped points, clamped at the ends. The
        clamping is why `coverage()` exists: a marker outside the mapped region
        gets the boundary value silently, and a segment lying entirely outside
        it would measure exactly 0 cM — a fabricated number, not a measurement.
        """
        group = self._by_chrom.get(int(chromosome))
        if group is None:
            raise KeyError(f"chromosome {chromosome} is not in the map")
        pos = group["position"].values
        positions = np.asarray(positions, dtype=float)
        return {
            "male": np.interp(positions, pos, group["male_cm"].values),
            "female": np.interp(positions, pos, group["female_cm"].values),
            "sexavg": np.interp(positions, pos, group["sexavg_cm"].values),
        }

    def coverage(self, chromosome, positions):
        """Fraction of `positions` falling outside the mapped region."""
        group = self._by_chrom[int(chromosome)]
        lo, hi = group["position"].min(), group["position"].max()
        positions = np.asarray(positions)
        outside = (positions < lo) | (positions > hi)
        return float(outside.mean())

    def annotate(self, markers, chromosome_col="chromosome", position_col="position"):
        """
        Attach `cm_male`, `cm_female` and `cm` to a marker table.

        `cm` is set to the sex-averaged map so that anything reading a single
        genetic coordinate — segment lengths, thinning, reporting — keeps
        working unchanged and stays comparable with published cM figures.
        """
        out = markers.copy()
        for col in ("cm_male", "cm_female", "cm"):
            out[col] = np.nan
        for chrom, group in out.groupby(chromosome_col):
            if int(chrom) not in self._by_chrom:
                continue
            cm = self.interpolate(chrom, group[position_col].values)
            out.loc[group.index, "cm_male"] = cm["male"]
            out.loc[group.index, "cm_female"] = cm["female"]
            out.loc[group.index, "cm"] = cm["sexavg"]
        return out.dropna(subset=["cm"]).reset_index(drop=True)

    def summary(self):
        totals = self.table.groupby("chromosome")[
            ["male_cm", "female_cm", "sexavg_cm"]
        ].max().sum()
        return {
            "rows": len(self.table),
            "chromosomes": len(self._by_chrom),
            "male_cm": round(float(totals.male_cm), 1),
            "female_cm": round(float(totals.female_cm), 1),
            "sexavg_cm": round(float(totals.sexavg_cm), 1),
            "male_crossovers_per_gamete": round(float(totals.male_cm) / 100, 1),
            "female_crossovers_per_gamete": round(float(totals.female_cm) / 100, 1),
        }


def fallback_scales():
    """
    Global rate scales for use when no sex-specific map is available.

    These reproduce the genome-wide totals exactly and nothing else. They are
    the weaker model, kept only so the tools run without the derived file.
    """
    return MALE_TOTAL_CM / SEXAVG_TOTAL_CM, FEMALE_TOTAL_CM / SEXAVG_TOTAL_CM
