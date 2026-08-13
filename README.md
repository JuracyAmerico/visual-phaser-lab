# visual-phaser-lab

Companion tooling for [visual-phaser](https://github.com/mickjolley/visual-phaser).
The upstream tool produces the pairwise comparison; this adds the layer that
turns it into inference, plus the machinery to know whether any of it is right.

**No genotypes live here.** Real kits are kept outside this repo entirely
(`~/Documents/DNA-Kits/`) and `.gitignore` blocks `*.csv` as a backstop.

## `vplab/simulate.py` — ground truth

Simulates 4 grandparents → 2 parents → N siblings over a real genetic map, then
writes vendor-format raw DNA alongside the *known* crossover positions and
grandparental origins. Every visual-phasing tool in existence asserts its
accuracy rather than measuring it, because real families do not come with known
crossovers. Simulated ones do.

Validated against the literature and against real kits:

| quantity | simulated | expected |
|----------|-----------|----------|
| IBD sharing between full siblings | 25.4 / 48.8 / 25.8 % | 25 / 50 / 25 |
| crossovers per gamete, paternal | 25.5 | 26.1 (map) |
| crossovers per gamete, maternal | 40.9 | 40.9 (map) |
| unrelated pair, fully identical per marker | 73.3 % | 74.0 % (measured on real kits) |
| unrelated pair, excluded per marker | 3.8 % | 3.3 % (measured on real kits) |

### Two ways this simulator was wrong, and how it was caught

Both were found by asking the simulator to reproduce a number measured on real
kits, rather than by reading its code.

**Allele frequencies belonged to the founder, not to the marker.** Every
founder drew a fresh reference/alternate pair, so each one effectively had a
private biallelic system and two unrelated people almost never shared an
allele. Measured: unrelated pairs came out *excluded* at 56% of markers where
real kits give 3.3%. Exclusion is the strongest signal visual phasing has, so
this handed the model roughly 17x more discriminating evidence than reality
supplies — the simulator was **easier** than real data, and this README
previously drew the opposite conclusion from the same observation, calling its
accuracy figures "pessimistic". They were overstatements.

**Interference was implemented by deleting crossovers.** Events falling within
10 cM of the previous one were dropped. That does suppress tight doubles, but
it also suppresses the rate: 37.4 maternal crossovers per gamete against a map
specifying 41. Ground truth was quietly wrong, so every count scored against it
was scored against the wrong target. Replaced with a gamma renewal process
(shape ν = 4), which is the standard model and keeps the mean rate at the map's
own while pushing the variance down.

**Remaining limitation:** founder haplotypes are still drawn independently per
marker, so there is no linkage disequilibrium. Marginal IBS statistics now
match real data, but LD structure does not — which is why marker thinning
cannot be chosen on simulated data (see below).

## `vplab/constraints.py` — the transitivity check

Grandparental origin is a two-valued attribute and equality on it is
transitive, so across three sibling pairs the number of paternal matches is 3
or 1, never 0 or 2. The only reachable (FIR, HIR, NIR) outcomes are (3,0,0),
(1,2,0), (1,0,2) and (0,2,1). This is upstream issue #4, decidable in O(1).

Verified: applied to true IBD state from the simulator it yields **exactly
zero** violations. Applied to raw per-SNP calls it also fires on
identity-by-state coincidence, so its proper home is segment-level calls —
which is what issue #4 actually describes.

Used as an oracle it found that **100% of impossible combinations in real
data, under both the original and corrected engine, sit on no-call loci** —
which is what exposed the no-call defect.

## `vplab/intake.py` — kit intake

Fingerprint, de-duplicate and place a raw DNA file, then relate it to every kit
already on file. Name-based filing is unsafe: Genera exports all arrive as
`dados_brutos<date>.csv.gz` with no identity inside, and two relatives in the
test family share a given name. One kit was mis-filed and only the DNA caught it.

    python -m vplab.intake <file.csv.gz> --name Alice --group other-relatives
    python -m vplab.intake --matrix

FIR is the discriminator: only people sharing *both* parents can be fully
identical over a segment, so it separates full siblings (~25%) from
half-siblings, aunts and grandparents (all ~0%) — which total cM cannot, since
all three sit near 1750 cM.

## `vplab/evaluate.py` and `vplab/calibration.py` — the measuring stick

`evaluate.py` scores the model against simulated truth: partition accuracy per
marker and per segment, crossover recovery per side, and reported-posterior vs
observed-accuracy. `calibration.py` fits the monotone map from raw score to
probability (isotonic regression, PAVA, ~15 lines of numpy — no scipy).

    python -m tools.evaluate_model --families 8      # every figure in this README
    python -m tools.phase_real --siblings A B C \
        --unrelated A M --parent-child A CHILD       # real kits, names via CLI

## `vplab/phase_hmm.py` — grandparental assignment

Lander–Green inheritance-vector HMM. Viterbi gives the assignment,
forward–backward gives a posterior per segment, and it takes N siblings rather
than the three every existing visual-phasing aid is limited to. 0.5s per
chromosome.

Emissions are calibrated from the data, not assumed — an unrelated pair gives
the IBD0 row, a parent–child pair gives IBD1 (they share exactly one copy at
every locus by definition), and apparent exclusions in that parent–child pair
give the error rate.

Measured on a real family dataset, per raw marker with no smoothing: a
parent–child pair is apparently excluded at **1.02%** of markers, and 64% of
those exclusions are isolated single markers — that is the genotyping error
rate the emission model needs, because the HMM consumes raw calls. (An earlier
0.0115% figure quoted here was the rate *after* the engine's isolated-mismatch
repair; both are correct measurements of different quantities, and the raw one
is the one this model is entitled to use.)

An unrelated pair reads as fully identical at **74% of individual markers**.
That figure is why visual phasing works on runs and never on single SNPs —
per-marker FIR carries almost no information, and the discriminating signal is
NIR (3.3% unrelated vs 1.0% parent–child).

## `vplab/genmap.py` — sex-specific genetic maps

Male and female meiosis do not just recombine at different *rates*, they
recombine in different *places*: male crossovers cluster near the telomeres and
are suppressed across the middle of a chromosome. Genome-wide the difference is
a factor of 1.57 (2608 cM against 4094 cM); locally the ratio swings from below
0.3 to above 3.

That distinction turned out to fix the model's worst failure — see below. The
map is not vendored (5.7 MB, no stated redistribution terms); build it with

    python -m tools.build_sex_map

from Bhérer, Campbell & Auton, *Refined genetic maps reveal sexual dimorphism in
human meiotic recombination at multiple scales*, Nat. Commun. **8**, 14994 (2017).

## Measured performance

Everything below comes from `python -m tools.evaluate_model` (8 simulated
families, 3 siblings each, 22 chromosomes) and `python -m tools.phase_real` on
two independent real trios. No figure here is quoted from memory; the rule
exists because the previous set of figures did not survive re-derivation.

| | before Phase 1 | after |
|---|---|---|
| segment-level partition accuracy | 66.5 % | **84.6 %** |
| marker-level accuracy | 85.5 %¹ | 79.5 % |
| paternal crossovers recovered | 0.62 | **0.96** |
| maternal crossovers recovered | 0.97 | 0.92 |
| chromosomes with the two sides transposed | 2 of 10 | **0 of 10** |
| calibration error, segment (leave-one-family-out) | not measured | 0.104 raw → **0.067** recalibrated |

¹ measured on the defective simulator described above, which was easier than
real data; the two marker figures are not comparable.

### The bug that mattered: whole chromosomes silently transposed

The paternal crossover count had been stuck at 56% of the biological rate on
real data with no explanation, and the standing hypothesis was that a
sex-averaged map distorts male recombination. **That hypothesis was wrong.** The
deficit reproduced on *simulated* data at 0.62 — where the model's map is
exactly the one that generated the data, so no map mismatch exists.

The real cause: with a sex-averaged map plus a global rate scale, the two
parental sides differ only in how *often* they switch. That is weak evidence,
each chromosome is phased independently, and so the paternal and maternal
labels came out transposed on 2 of 10 chromosomes. A transposed chromosome
assigns every one of its segments to the wrong grandparental couple, and
nothing in the output looks unusual.

Driving each side by its own genetic map — shape, not just rate — removed it
entirely (0 of 10) and lifted segment accuracy from 83% to 88% in the same run.

### Marker thinning was hiding the same defect

Thinning to 0.2 cM had been calibrated by pushing the spacing up until inferred
crossover counts stopped exploding. With the transition model corrected, that
crutch is no longer needed. Ratios to the literature rate, both sides, on two
independent real trios:

| thinning | paternal trio | maternal trio |
|---|---|---|
| unthinned | 5.6× | — |
| 0.02 cM | 1.11 / 0.92 | 1.14 / 1.16 |
| **0.08 cM** | **0.98 / 0.84** | **0.98 / 1.07** |
| 0.20 cM (old default) | 0.81 / 0.70 | — |

0.05–0.10 cM is a plateau; differences inside it are within the sampling noise
of ~78 crossovers (±11%). Note that thinning **cannot** be chosen on simulated
data: with no LD there, less thinning is always better. It is calibrated
against an external biological constant instead.

### Calibration: the plot no tool in this field publishes

A posterior is only useful if it means what it says. Raw segment posteriors are
consistently **under**-confident — segments scored 0.61 are right 76% of the
time — because a segment call is a majority vote across many markers, and a
vote is more reliable than the average of its voters' confidences. Isotonic
recalibration fitted on other families reduces the expected calibration error
from 0.104 to 0.067, scored strictly leave-one-family-out.

It does not reach zero, and the reason is measurable:

    mean posterior bias      -0.068   removable by calibration
    between-family spread     0.048   NOT removable by any global calibrator

Accuracy genuinely varies between families (78.5%–90.0%), so about 5 points of
calibration error is irreducible without run-level information. That sets the
target for later work rather than being papered over.

### The decision rule

`phase_hmm.assign_side()` turns segment votes into a verdict, or declines:

- **Votes on one chromosome count once.** The failure mode that matters flips a
  whole chromosome at a time, so ten segments from one chromosome are one piece
  of evidence, not ten.
- **Fewer than three chromosomes is not a verdict.** At ~85% per-segment
  accuracy a single segment still looks perfectly clean — a single segment
  cannot contradict itself. This is how a set of confident, mutually
  inconsistent line assignments got produced here once.

### Real data: crossover counts on two independent trios

Human meiosis is tightly constrained, so a crossover count far off the
biological rate condemns the transition model before any segment call is worth
reading. Ratios to literature (~26.1 male, ~40.9 female per gamete):

| trio | side A | side B |
|---|---|---|
| three full siblings, family 1 | 0.98 | 0.84 |
| three full siblings, family 2 | 0.98 | 1.07 |

Three of the four sit within 7% of the literature rate; the fourth is at 0.84
against a ±15% target, and the sampling noise on ~78 crossovers is about ±11%.
Before Phase 1 the same measurement gave 0.56 on one side.

Sides are labelled A and B on purpose. **Which one is the father's is not
established by these counts** — the model is told one side recombines less, so
reading the answer off the rate would be circular. Naming a side takes an
outside anchor: a relative descending from exactly one grandparent.

### Real data, via an independent anchor

Where the HMM says two siblings inherited the same maternal haplotype, their
matching to a great-aunt is 86.5% concordant; where it says they differ, 66.9%.
A **+19.6% lift** against evidence the model never saw.

### Two validation designs that failed first

Both are recorded because each looked reasonable and produced a confident
wrong answer.

1. **Saturation.** Testing against "matches any of three great-aunts" — each
   shares ~50% of the genome, so the union covers ~87% and there is no contrast
   left to measure. Produced a flat 80–90% on both groups.

2. **A pedigree-level error.** A great-aunt is the sister of the trio's
   *mother*, so she descends from *both* of that mother's parents and is
   related to both maternal groups. She cannot discriminate which grandparent a
   segment came from, and the near-zero separation measured was the correct
   answer to a meaningless question. Anchoring *which* grandparent needs someone
   related to only one of them.

## Not yet built

- Modelling linkage disequilibrium instead of thinning past it
- An empirical regional null from the user's own calibration pairs
- Pile-up masking (HLA and friends)
- The match layer: line assignment as a joint model rather than per-window votes,
  anchor propagation, and "who should I test next"
