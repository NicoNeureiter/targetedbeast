"""Paired comparison of benchmark settings, rep against rep (same seed), from a summary TSV.

Prints the ESS gate per cell, then for each setting against the baseline: mean difference in the
metric, its t-based 90% CI (df = n - 1) and the difference relative to the baseline mean.

Usage:
    python scripts/paired_compare.py runs/infb/thorough/aos_baseline.summary.tsv \
        --baseline pca_grid_mid --settings pca pca_aos_mid
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Two-sided 90% (upper 0.95 quantile) Student t critical values; no scipy in this environment.
T_95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860,
        9: 1.833, 10: 1.812}

ESS_GATE = 100.0
ESS_COLUMNS = ("posterior.ESS", "tree.height.ESS")


def read_cells(path: Path, metric: str) -> dict[str, dict[int, dict]]:
    cells: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cells[row["setting"]][int(row["rep"])] = {
                "seed": row["seed"],
                "metric": float(row[metric]),
                **{c: float(row.get(c) or "nan") for c in ESS_COLUMNS},
            }
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--metric", default="esjdPerSec_rnni")
    args = parser.parse_args()

    cells = read_cells(args.summary, args.metric)
    for setting in [args.baseline, *args.settings]:
        if setting not in cells:
            raise SystemExit(f"{args.summary} has no rows for setting {setting!r}")

    print("setting\trep\tseed\t" + "\t".join(ESS_COLUMNS) + f"\t{args.metric}\tgate")
    for setting in [args.baseline, *args.settings]:
        for rep, cell in sorted(cells[setting].items()):
            ok = all(cell[c] >= ESS_GATE for c in ESS_COLUMNS)
            print(f"{setting}\t{rep}\t{cell['seed']}\t"
                  + "\t".join(f"{cell[c]:.1f}" for c in ESS_COLUMNS)
                  + f"\t{cell['metric']:.1f}\t{'ok' if ok else 'FAIL'}")
    print()

    base = cells[args.baseline]
    print(f"setting\tn\tmean diff\t90% CI\texcludes 0\tvs {args.baseline}")
    for setting in args.settings:
        reps = sorted(set(base) & set(cells[setting]))
        diffs = [cells[setting][r]["metric"] - base[r]["metric"] for r in reps]
        n = len(diffs)
        if n < 2:
            print(f"{setting}\t{n}\t(need at least 2 paired reps)")
            continue
        mean = statistics.fmean(diffs)
        half = T_95[n - 1] * statistics.stdev(diffs) / math.sqrt(n)
        lo, hi = mean - half, mean + half
        rel = mean / statistics.fmean(base[r]["metric"] for r in reps)
        print(f"{setting}\t{n}\t{mean:+.1f}\t({lo:+.1f}, {hi:+.1f})\t"
              f"{'yes' if lo > 0 or hi < 0 else 'no'}\t{rel:+.1%}")


if __name__ == "__main__":
    main()
