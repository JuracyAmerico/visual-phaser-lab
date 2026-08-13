# -*- coding: utf-8 -*-
"""
Regression tests for the properties Phase 1 established.

Each of these guards a defect that was actually present and actually measured,
not a hypothetical one. They are cheap by design — a few chromosomes, a few
thousand markers — so the full file runs in seconds and can sit in front of
every commit.
"""
import numpy as np
import pandas as pd
import pytest

from vplab import calibration as C
from vplab import evaluate as E
from vplab import genmap as G
from vplab import phase_hmm as H
from vplab import simulate as S

MAP_PATH = "data/min_map.txt"


@pytest.fixture(scope="module")
def sex_map():
    m = G.SexSpecificMap.load()
    if m is None:
        pytest.skip("sex-specific map not built; run `python -m tools.build_sex_map`")
    return m


@pytest.fixture(scope="module")
def panel(sex_map):
    gmap = S.load_map(MAP_PATH, chromosomes=[1, 2, 6, 21])
    return S.build_marker_panel(gmap, n_markers_per_chrom=8000,
                                rng=np.random.default_rng(3), sex_map=sex_map)


class TestSimulatorRealism:
    """The simulator is the measuring stick; a biased stick is worse than none."""

    def test_unrelated_pair_matches_real_kit_statistics(self, panel):
        """
        Two unrelated people must coincide about as often as they really do.

        Allele identity is a property of the marker, not of the individual. An
        earlier version drew a fresh reference/alternate pair per founder,
        giving each one a private biallelic system: unrelated people then
        looked *excluded* at 56% of markers where the reference dataset sits
        at 3.3%. Exclusion is the strongest signal the method has, so that made
        simulated data far easier than reality and every accuracy measured on
        it an overstatement.
        """
        ped = S.simulate_pedigree(panel, n_siblings=2, seed=5)
        calls = E.match_calls(ped.relatives["Unrelated"], ped.genotypes["Sib1"])
        informative = calls[calls != E.MISSING]
        fir = float(np.mean(informative == E.FIR))
        nir = float(np.mean(informative == E.NIR))
        assert 0.68 <= fir <= 0.80, (
            f"unrelated FIR {fir:.1%}, reference dataset gives 74.0%")
        assert 0.02 <= nir <= 0.06, (
            f"unrelated NIR {nir:.1%}, reference dataset gives 3.3%")

    def test_parent_child_pair_is_never_excluded(self, panel):
        """A parent and child share one copy at every locus, by definition."""
        ped = S.simulate_pedigree(panel, n_siblings=1, seed=5)
        calls = E.match_calls(ped.relatives["Father"], ped.genotypes["Sib1"])
        informative = calls[calls != E.MISSING]
        assert float(np.mean(informative == E.NIR)) == 0.0

    def test_crossover_rate_matches_the_map(self, panel):
        """
        Interference must suppress tight doubles without suppressing the rate.

        The previous implementation drew Poisson crossovers then deleted any
        within 10 cM of the previous one. That does thin out close pairs, but
        it also removes events outright: measured, 37.4 maternal crossovers per
        gamete where the map specifies 41. Ground truth was quietly wrong, so
        every count scored against it was scored against the wrong target.
        """
        rng = np.random.default_rng(9)
        for sex, column in (("male", "cm_male"), ("female", "cm_female")):
            expected = sum(
                g[column].max() - g[column].min()
                for _, g in panel.groupby("chromosome")
            ) / 100.0
            counts = [
                sum(len(v) for v in S._simulate_meiosis(panel, rng, sex=sex)[1].values())
                for _ in range(40)
            ]
            observed = float(np.mean(counts))
            assert observed == pytest.approx(expected, rel=0.15), (
                f"{sex}: {observed:.1f} crossovers/gamete against a map "
                f"specifying {expected:.1f}"
            )


class TestSideIdentifiability:
    """The failure that corrupted whole chromosomes without looking wrong."""

    def test_sex_specific_maps_keep_the_two_sides_apart(self, panel):
        """
        With only a global rate scale, which side is which is barely
        identifiable inside one chromosome, and chromosomes are phased
        independently — so the paternal and maternal labels came out
        transposed on 2 of 10 simulated chromosomes. A transposed chromosome
        assigns every one of its segments to the wrong grandparental couple,
        and nothing in the output looks unusual.

        Male and female maps differ in shape, not just total length, which is
        local positional evidence rather than a weak global prior.
        """
        ped = S.simulate_pedigree(panel, n_siblings=3, seed=17,
                                  no_call_rate=0.002, error_rate=0.0002)
        emissions = E.calibrate_from_pedigree(ped)
        results, _, siblings = E.phase_pedigree(ped, emissions=emissions)

        for res in results:
            truth_pat = E.canonical_partition(
                E.truth_bits(ped, siblings, "paternal", res.keep))
            truth_mat = E.canonical_partition(
                E.truth_bits(ped, siblings, "maternal", res.keep))
            called = res.called_partition(0)
            as_paternal = np.mean([a == b for a, b in zip(called, truth_pat)])
            as_maternal = np.mean([a == b for a, b in zip(called, truth_mat)])
            assert as_paternal >= as_maternal, (
                f"chr{res.chromosome}: side 0 tracks the true *maternal* "
                f"partition ({as_maternal:.2f}) better than the paternal one "
                f"({as_paternal:.2f}) — the two sides are transposed"
            )


class TestDecisionRule:
    """assign_side must decline rather than guess."""

    def test_one_chromosome_is_never_a_verdict(self):
        """
        A single segment cannot contradict itself, so at ~85% per-segment
        accuracy it looks perfectly clean while being wrong one time in seven.
        That is exactly how a set of confident, mutually inconsistent line
        assignments got produced here once.
        """
        call = H.assign_side([(1, "P0", 0.97), (1, "P0", 0.96), (1, "P0", 0.95)])
        assert call.call == H.INSUFFICIENT
        assert call.n_chromosomes == 1

    def test_three_chromosomes_agreeing_is_a_verdict(self):
        call = H.assign_side([(1, "P0", 0.9), (5, "P0", 0.9), (12, "P0", 0.9)])
        assert call.call == "P0"
        assert call.probability > 0.99
        assert call.support == 1.0

    def test_votes_on_one_chromosome_count_once(self):
        """
        Segments on one chromosome share a failure mode — the whole chromosome
        transposing — so they are one piece of evidence, not many.
        """
        many = [(1, "P0", 0.9)] * 8 + [(4, "P1", 0.9), (9, "P1", 0.9)]
        call = H.assign_side(many, min_chromosomes=3)
        assert call.n_chromosomes == 3
        assert call.call == "P1", "eight correlated votes outvoted two independent ones"

    def test_disagreement_lowers_confidence(self):
        agree = H.assign_side([(1, "P0", 0.9), (2, "P0", 0.9), (3, "P0", 0.9)])
        split = H.assign_side([(1, "P0", 0.9), (2, "P0", 0.9), (3, "P1", 0.9)])
        assert split.probability < agree.probability
        assert split.support == pytest.approx(2 / 3)


class TestCalibration:
    def test_isotonic_fit_is_monotone_and_unbiased(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.25, 1.0, 20000)
        # Scores that systematically understate accuracy, as measured.
        truth = np.clip(raw + 0.12, 0, 1)
        correct = (rng.random(20000) < truth).astype(float)
        cal = C.IsotonicCalibrator.fit(raw, correct)
        assert np.all(np.diff(cal.y) >= -1e-9), "calibration map is not monotone"
        before = C.expected_calibration_error(raw, correct)
        after = C.expected_calibration_error(cal.transform(raw), correct)
        assert after < before / 2, f"ECE {before:.3f} -> {after:.3f}"

    def test_partition_posterior_sums_to_one(self, panel):
        ped = S.simulate_pedigree(panel, n_siblings=3, seed=21)
        emissions = E.calibrate_from_pedigree(ped)
        results, _, _ = E.phase_pedigree(ped, emissions=emissions, chromosomes=[21])
        for res in results:
            for side in (0, 1):
                _, probs = res.partition_posterior(side)
                assert np.allclose(probs.sum(axis=1), 1.0)
