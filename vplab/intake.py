# -*- coding: utf-8 -*-
"""
Kit intake — fingerprint, de-duplicate, and place a raw DNA file, then report
how it relates to every kit already on file.

Built because name-based filing is unsafe here: every Genera export arrives as
`dados_brutos<date>.csv.gz` with no identity inside the file, and two relatives
in this family share a given name. One kit was already mis-filed and only the
DNA caught it. So nothing is trusted from a filename — identity is established
from the data, every time.

Checks performed, in order:
  1. MD5 against every filed kit          -- catches the same file sent twice
  2. SNP count, genome build, sex, no-call rate
  3. Total HIR and FIR sharing against every filed kit

FIR is the discriminator that matters. Only people who share BOTH parents can
be fully identical over a segment, so FIR separates full siblings (~25%) from
half-siblings, aunts/uncles and grandparents (all ~0%) -- which total cM alone
cannot do, since all three sit near 1750 cM.

Usage:
    python -m vplab.intake <file.csv[.gz]> --name Maria --group other-relatives
    python -m vplab.intake --matrix          # re-report across all filed kits
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KITS_DIR = Path.home() / "Documents" / "DNA-Kits"
ENGINE = Path("/Users/americo/projects/visual-phaser/Visual_Phaser.V1.2.py")
MAP = Path(__file__).resolve().parent.parent / "data" / "min_map.txt"

# Autosomal genetic length, sex-averaged, used to express FIR as a fraction.
GENOME_CM = 3545.0

# rs3131972 is an early chr1 marker present on essentially every consumer chip.
BUILD_ANCHOR = ("rs3131972", 752721, 817341)   # rsid, GRCh37 pos, GRCh38 pos


def _load_engine():
    """Import the comparison engine (its filename contains dots)."""
    import importlib.util
    import os

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    os.environ.pop("VP_CONFIG_PATH", None)
    sys.path.insert(0, str(ENGINE.parent))
    try:
        spec = importlib.util.spec_from_file_location("vp_engine", ENGINE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved_argv


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_kit(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return pd.read_csv(handle, dtype=str)


def fingerprint(path):
    """Identity facts derivable from the file itself, with no reliance on its name."""
    frame = read_kit(path)
    x = frame[frame["CHROMOSOME"] == "X"]
    result = x["RESULT"].fillna("")
    het = (result.str.len().eq(2) & (result.str[0] != result.str[1])).mean()

    rsid, b37, b38 = BUILD_ANCHOR
    hit = frame[frame["RSID"] == rsid]
    pos = int(hit["POSITION"].iloc[0]) if len(hit) else None
    build = "GRCh37" if pos == b37 else ("GRCh38" if pos == b38 else f"unknown ({pos})")

    no_call = frame["RESULT"].fillna("--").str.contains("-").mean()

    return {
        "snps": len(frame),
        "build": build,
        # Males carry one X, so heterozygosity is near zero apart from the
        # pseudoautosomal region; females run 20-30%.
        "sex": "M" if het < 0.05 else "F",
        "x_het": het,
        "no_call": no_call,
    }


def filed_kits():
    return sorted(KITS_DIR.glob("*/*.csv"))


def relate(engine, gmap, dna_a, dna_b, chromosomes=range(1, 23)):
    """Total HIR and FIR shared, in cM, using the engine's own segment caller."""
    hir_cm = fir_cm = 0.0
    for chrom in chromosomes:
        chrom_map = gmap[gmap["Chromosome"] == chrom].sort_values("Position")
        if chrom_map.empty:
            continue
        a = dna_a[dna_a["chromosome"] == chrom]
        b = dna_b[dna_b["chromosome"] == chrom]
        merged = a.merge(b, on=("rsid", "chromosome", "position"), suffixes=("_1", "_2")).copy()
        if merged.empty:
            continue
        merged["match"] = engine.apply_conditions_vectorized(
            merged["allele1_1"].values, merged["allele2_1"].values,
            merged["allele1_2"].values, merged["allele2_2"].values, "X",
        )
        merged = engine.repair_files_optimized(merged, 75, 1000)
        hir, fir = engine.scan_genomes_optimized(
            merged, chrom, 7, 1, 200, 75, 1000,
            chrom_map["Position"].values, chrom_map["cM"].values,
        )
        hir_cm += hir["Length (cM)"].sum() if len(hir) else 0.0
        fir_cm += fir["Length (cM)"].sum() if len(fir) else 0.0
    return hir_cm, fir_cm


def interpret(hir_cm, fir_pct):
    """
    Name the relationship. FIR does the work: without shared FIR the pair
    cannot share both parents, whatever the total says.
    """
    if hir_cm > 3400 and fir_pct < 2:
        return "PARENT-CHILD (whole genome half-identical, no FIR)"
    if hir_cm > 2000 and fir_pct > 10:
        return "FULL SIBLINGS"
    if hir_cm > 1200:
        return "aunt/uncle, half-sibling or grandparent (zero FIR — NOT full siblings)"
    if hir_cm > 200:
        return "distant relative"
    return "unrelated"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="raw DNA file to take in")
    parser.add_argument("--name", help="short kit name, e.g. Maria")
    parser.add_argument("--group", default="other-relatives",
                        help="subfolder: paternal-trio | maternal-trio | other-relatives")
    parser.add_argument("--matrix", action="store_true",
                        help="report relationships across all filed kits and exit")
    args = parser.parse_args(argv)

    engine = _load_engine()
    gmap = pd.read_csv(MAP, sep="\t")

    if args.matrix:
        kits = filed_kits()
        loaded = {p.name.split("_raw_dna")[0]: engine.agnostic_load_individual_dna(
            p.name.split("_raw_dna")[0], str(p.parent), "X")[1] for p in kits}
        import itertools
        rows = []
        for a, b in itertools.combinations(sorted(loaded), 2):
            h, f = relate(engine, gmap, loaded[a], loaded[b])
            rows.append((a, b, h, f / GENOME_CM * 100))
        for a, b, h, p in sorted(rows, key=lambda r: -r[2]):
            print(f"{a+'-'+b:42} {h:8.1f} cM  FIR {p:5.1f}%  {interpret(h, p)}")
        return 0

    if not args.path or not args.name:
        parser.error("a file path and --name are required unless --matrix is used")

    src = Path(args.path).expanduser()
    print(f"=== intake: {src.name} ===")

    # 1. Duplicate check against everything already filed.
    tmp = src
    if src.suffix == ".gz":
        tmp = Path("/tmp") / src.stem
        with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    digest = md5(tmp)
    for existing in filed_kits():
        if md5(existing) == digest:
            print(f"  REJECTED — byte-identical to {existing.name}")
            print("  This is the same file sent twice, not a new person.")
            return 1
    print(f"  md5 {digest}  (no duplicate on file)")

    # 2. Identity facts from the data.
    fp = fingerprint(tmp)
    print(f"  {fp['snps']:,} SNPs | {fp['build']} | sex {fp['sex']} "
          f"(X het {fp['x_het']:.1%}) | no-call {fp['no_call']:.2%}")
    if fp["build"] != "GRCh37":
        print("  WARNING: not GRCh37 — the genetic map and every comparison assume build 37")

    # 3. Place it.
    dest = KITS_DIR / args.group / f"{args.name}_raw_dna.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp, dest)
    print(f"  filed as {dest.relative_to(KITS_DIR)}")

    # 4. Relate it to everyone already on file.
    print("\n=== relationships ===")
    new = engine.agnostic_load_individual_dna(args.name, str(dest.parent), "X")[1]
    results = []
    for other in filed_kits():
        name = other.name.split("_raw_dna")[0]
        if name == args.name:
            continue
        dna = engine.agnostic_load_individual_dna(name, str(other.parent), "X")[1]
        h, f = relate(engine, gmap, new, dna)
        results.append((name, h, f / GENOME_CM * 100))
    for name, h, p in sorted(results, key=lambda r: -r[1]):
        print(f"  {args.name}-{name:24} {h:8.1f} cM  FIR {p:5.1f}%  {interpret(h, p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
