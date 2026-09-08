"""Collect MCMC run statistics for a single BenchmarkMCMC replicate (one dataset/tier/setting/rep
cell of the evaluation pipeline - targetedbeast-evaluation-pipeline.md Phase 5).

Combines:
  * the cell's `run.opstats.tsv`  -> BenchmarkMCMC's own per-operator accept/reject/timing/ESJD
                                      rows plus the chain-level `__CHAIN__` headline row. This
                                      supersedes parsing the `.state` file's operator JSON comment
                                      (the pre-BenchmarkMCMC approach): opstats.tsv already has
                                      acceptRate and ESJD computed, so there is nothing left to
                                      derive from `.state` for this pipeline.
  * loganalyser `-oneline` output (optional - only present for ESS-gated tiers) -> per-parameter
    ESS, reused via the same lowest-ESS-among-model-parameters logic as before.
  * a `run.runtime` file -> wall-clock runtime in seconds.
  * (optional) the cell's `run.trees` -> tree-space pseudo-ESS (Lanfear et al. 2016), via
    `tree_pseudo_ess.py`. Empirically much more stable than loganalyser's per-parameter ESS
    (see docs/evaluation/where-targeting-wins.md) and closer to what actually gates whether a
    chain has explored tree space adequately, rather than just one scalar summary of it.

and writes one summary row per cell: dataset/tier/setting/rep/seed, the chain-level headline
columns from opstats.tsv, and (if given) the ESS columns.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

# tree_pseudo_ess.py needs TreeTracer's own venv (rapidtrees is a compiled extension installed
# there, not in this project's venv) - see that script's docstring.
TREETRACER_PYTHON = Path("/Users/nneureiter/Projects/treetracer/.venv/bin/python")
TREE_PSEUDO_ESS_SCRIPT = Path(__file__).resolve().parent / "tree_pseudo_ess.py"


def compute_tree_pseudo_ess(trees_path: Path, burnin_percent: float) -> dict:
    """Shell out to tree_pseudo_ess.py (needs TreeTracer's venv) and parse its JSON line."""
    out = subprocess.run(
        [str(TREETRACER_PYTHON), str(TREE_PSEUDO_ESS_SCRIPT), str(trees_path),
         "--burnin-percent", str(burnin_percent)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])

CHAIN_ROW_NAME = "__CHAIN__"


def read_runtime_seconds(path: Path) -> float:
    """Read wall-clock runtime (seconds) from a one-number runtime file."""
    return float(path.read_text().strip())


def read_tsv(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty TSV: {path}")
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def read_loganalyser_oneline(path: Path) -> dict[str, str]:
    """Parse a loganalyser `-oneline` file (header row + single value row)."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Unexpected loganalyser output in {path}")
    header = lines[0].split("\t")
    values = lines[1].split("\t")
    return dict(zip(header, values))


# Logged quantities that are tuning diagnostics rather than model parameters. Their ESS says
# nothing about how well the chain explored the posterior: an adapting coefficient is a slowly
# drifting non-stationary trace by construction, so its ESS is always poor and would dominate
# the "lowest ESS" summary, making every transform run look worse than its baseline.
NON_PARAMETER_PREFIXES = ("targetWeightTransform.", "edgeWeightTransform.")


def is_model_parameter(param: str) -> bool:
    """Whether a logged column should count toward the lowest-ESS summary."""
    return not param.startswith(NON_PARAMETER_PREFIXES)


def lowest_ess(row: dict[str, str]) -> tuple[str, float]:
    """Return (parameter, value) for the smallest finite ESS among the model parameters."""
    best: tuple[str, float] | None = None
    for col, val in row.items():
        if not col.endswith(".ESS"):
            continue
        if not is_model_parameter(col[: -len(".ESS")]):
            continue
        try:
            ess = float(val)
        except ValueError:
            continue
        if not math.isfinite(ess):
            continue
        param = col[: -len(".ESS")]
        if best is None or ess < best[1]:
            best = (param, ess)
    if best is None:
        raise ValueError("No finite ESS values found in loganalyser output")
    return best


def chain_row(opstats_rows: list[dict[str, str]]) -> dict[str, str]:
    for row in opstats_rows:
        if row.get("operator") == CHAIN_ROW_NAME:
            return row
    raise ValueError(f"No {CHAIN_ROW_NAME} row found in opstats")


def fmt(x) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opstats", type=Path, required=True, help="run.opstats.tsv")
    parser.add_argument("--runtime", type=Path, required=True, help="run.runtime file (seconds)")
    parser.add_argument("--ess-table", type=Path, default=None,
                         help="loganalyser -oneline output; omitted for non-ESS-gated tiers")
    parser.add_argument("--trees", type=Path, default=None,
                         help="run.trees file; when given, adds tree-space pseudo-ESS columns "
                              "(treeESS_min/median/max) via TreeTracer's pseudo-ESS estimator")
    parser.add_argument("--tree-burnin-percent", type=float, default=10.0,
                         help="Burn-in percent for --trees, by tree count. Default 10 (matches "
                              "the loganalyser -b default used for --ess-table).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--rep", required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    runtime_sec = read_runtime_seconds(args.runtime)
    runtime_hours = runtime_sec / 3600.0

    chain = chain_row(read_tsv(args.opstats))

    row = {
        "dataset": args.dataset,
        "tier": args.tier,
        "setting": args.setting,
        "rep": args.rep,
        "seed": chain.get("seed", "NA"),
        "runtime_sec": fmt(runtime_sec),
        "runtime_hours": fmt(runtime_hours),
        "proposals": chain.get("proposals", "NA"),
        "acceptRate": chain.get("acceptRate", "NA"),
        "meanJumpGivenAccepted": chain.get("meanJumpGivenAccepted", "NA"),
    }
    # esjdPerSec_<metric> columns (typically esjdPerSec_rnni) - carried through verbatim,
    # whichever metrics the cell's run.xml declared (empty for the primates smoke tier, which
    # injects no metric at all - those columns are simply absent from opstats.tsv there).
    for col, val in chain.items():
        if col.startswith("esjdPerSec_") or col.startswith("esjd_"):
            row[col] = val

    if args.ess_table is not None:
        log_row = read_loganalyser_oneline(args.ess_table)
        low_param, low_ess = lowest_ess(log_row)
        ess_per_hour = low_ess / runtime_hours if runtime_hours > 0 else float("nan")
        row["lowest_ESS"] = fmt(low_ess)
        row["lowest_ESS_param"] = low_param
        row["ESS_per_hour"] = fmt(ess_per_hour)
        for col, val in log_row.items():
            if col.endswith(".ESS"):
                row[col] = val

    if args.trees is not None:
        tree_ess = compute_tree_pseudo_ess(args.trees, args.tree_burnin_percent)
        row["treeESS_min"] = fmt(tree_ess["min"])
        row["treeESS_median"] = fmt(tree_ess["median"])
        row["treeESS_max"] = fmt(tree_ess["max"])
        row["treeESS_n_trees_used"] = fmt(tree_ess["n_trees_used"])

    header = list(row.keys())
    values = [row[k] for k in header]
    args.summary_out.write_text("\t".join(header) + "\n" + "\t".join(values) + "\n")

    print(
        f"{args.dataset}/{args.tier}/{args.setting}/rep{args.rep}: "
        f"runtime={runtime_sec:.0f}s acceptRate={row['acceptRate']} "
        f"meanJumpGivenAccepted={row['meanJumpGivenAccepted']}"
        + (f" lowest_ESS={row.get('lowest_ESS')} ({row.get('lowest_ESS_param')})"
           if args.ess_table is not None else "")
        + (f" treeESS_min={row.get('treeESS_min')} treeESS_median={row.get('treeESS_median')}"
           if args.trees is not None else "")
    )


if __name__ == "__main__":
    main()
