"""Compute a reference log-posterior band from a completed BEAST run.log (Phase 3c of
targetedbeast-evaluation-pipeline.md).

The burn-in-speed metric (BurnInSpeedTracker, wired into BenchmarkMCMC via
referenceLowerBound/referenceUpperBound) needs the 5th-95th percentile of log-posterior over the
post-burn-in portion of a converged reference chain, computed once per dataset from a neutral
baseline setting - not per-comparison, and not at per-proposal resolution: the tree-log's own
"posterior" column, logged at BEAST's normal log interval, is an adequate (if coarser) proxy for
computing the band itself, per the task's design note. Only the *measurement* against the band
needs proposal-level precision, and BurnInSpeedTracker already provides that internally.

Percentile method mirrors BurnInSpeedSpecTest's percentile() exactly (ceil(p*n)-1, clamped) so the
Java test's own reference-band derivation and this script agree on the same definition.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_posterior_column(log_path: Path) -> list[float]:
    """Reads the "posterior" column of a BEAST trace log, by header name (not fixed index) so
    this works across datasets whose logged columns differ (primates vs infB vs iecor)."""
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise SystemExit(f"No trace rows found in {log_path}")
    header = lines[0].split("\t")
    if "posterior" not in header:
        raise SystemExit(f"{log_path} has no 'posterior' column (header: {header})")
    col = header.index("posterior")
    values = []
    for ln in lines[1:]:
        cols = ln.split("\t")
        values.append(float(cols[col]))
    return values


def percentile(sorted_values: list[float], p: float) -> float:
    index = math.ceil(p * len(sorted_values)) - 1
    index = max(0, min(len(sorted_values) - 1, index))
    return sorted_values[index]


def compute_band(log_path: Path, burnin_percent: float) -> dict:
    values = read_posterior_column(log_path)
    discard = int(len(values) * burnin_percent / 100.0)
    post_burnin = values[discard:]
    if len(post_burnin) < 2:
        raise SystemExit(
            f"Only {len(post_burnin)} post-burnin samples in {log_path} (burnin={burnin_percent}%, "
            f"{len(values)} total) - not enough to compute a reference band"
        )
    sorted_post_burnin = sorted(post_burnin)
    lower = percentile(sorted_post_burnin, 0.05)
    upper = percentile(sorted_post_burnin, 0.95)
    if not lower < upper:
        raise SystemExit(f"Degenerate reference band [{lower}, {upper}] from {log_path}")
    return {
        "lower": lower,
        "upper": upper,
        "source_log": str(log_path),
        "burnin_percent": burnin_percent,
        "total_samples": len(values),
        "post_burnin_samples": len(post_burnin),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="reference run's run.log")
    parser.add_argument("--burnin", type=float, default=10,
                         help="burn-in discard fraction as a percentage, matching "
                              "'loganalyser -b' throughout this project (default: 10)")
    parser.add_argument("--out", type=Path, required=True, help="output reference_band.json path")
    args = parser.parse_args()

    band = compute_band(args.log, args.burnin)
    args.out.write_text(json.dumps(band, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}: lower={band['lower']:.4f} upper={band['upper']:.4f} "
          f"({band['post_burnin_samples']} post-burnin samples)")


if __name__ == "__main__":
    main()
