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

Validated against the literature:

| quantity | simulated | expected |
|----------|-----------|----------|
| IBD sharing between full siblings | 25.4 / 48.8 / 25.8 % | 25 / 50 / 25 |
| crossovers per gamete, paternal | 28.2 | ~26 |
| crossovers per gamete, maternal | 41.8 | ~42 |

**Known limitation:** founder haplotypes are drawn per-marker with no linkage
disequilibrium, so identity-by-state statistics are not realistic — simulated
data shows far more IBS/IBD discordance than real data. Crossover and IBD
behaviour are sound; do not trust IBS-level numbers from it.

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
`dados_brutos<date>.csv.gz` with no identity inside, and two relatives in this
family share a given name. One kit was mis-filed and only the DNA caught it.

    python -m vplab.intake <file.csv.gz> --name Maria --group other-relatives
    python -m vplab.intake --matrix

FIR is the discriminator: only people sharing *both* parents can be fully
identical over a segment, so it separates full siblings (~25%) from
half-siblings, aunts and grandparents (all ~0%) — which total cM cannot, since
all three sit near 1750 cM.

## Not yet built

- Segment export (CSV / JSON / DNA Painter format)
- The Lander–Green inheritance-vector HMM: grandparental assignment with
  per-segment posterior confidence, for N siblings rather than exactly 3
