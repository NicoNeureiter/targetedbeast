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
  * (optional) the cell's `run.trees` -> tree-space pseudo-ESS (Lanfear et al. 2016) for RF and
    KF distance, via `tree_pseudo_ess.py`.

and writes one summary row per cell: dataset/tier/setting/rep/seed, the chain-level headline
columns from opstats.tsv, and (if given) the ESS columns.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

CHAIN_ROW_NAME = "__CHAIN__"

# tree_pseudo_ess.py needs TreeTracer's venv: rapidtrees is a compiled extension installed there.
TREETRACER_PYTHON = Path("/Users/nneureiter/Projects/treetracer/.venv/bin/python")
TREE_PSEUDO_ESS_SCRIPT = Path(__file__).resolve().parent / "tree_pseudo_ess.py"


def compute_tree_pseudo_ess(trees_path: Path, burnin_percent: float) -> dict:
    """Run tree_pseudo_ess.py under TreeTracer's venv and parse its JSON line."""
    out = subprocess.run(
        [str(TREETRACER_PYTHON), str(TREE_PSEUDO_ESS_SCRIPT), str(trees_path),
         "--burnin-percent", str(burnin_percent)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


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


def operator_id(name: str) -> str:
    """`pkg.Class(id)` -> `id`."""
    return name[name.rfind("(") + 1:-1] if name.endswith(")") else name


def targeted_metrics(opstats_rows: list[dict[str, str]], targeted: set[str]) -> dict[str, float]:
    """ESJD per proposal and per second over the targeted operators only, for each metric.

    A weight transform only changes these operators' proposals, and on infB they contribute ~13%
    of chain-level ESJD, so the chain-level figure dilutes their effect. An operator counts if its
    id, or the part before its first '.', is listed - `targetedWilsonBalding.e2`, a candidate
    inside an AdaptableOperatorSampler, counts as `targetedWilsonBalding`.
    """
    rows = [r for r in opstats_rows if r.get("operator") != CHAIN_ROW_NAME
            and operator_id(r["operator"]).split(".")[0] in targeted]
    if not rows:
        raise ValueError(f"No opstats rows for targeted operators {sorted(targeted)}")
    proposals = sum(float(r["proposals"]) for r in rows)
    seconds = sum(float(r["proposals"]) * float(r["meanTimeMs"]) for r in rows) / 1000.0
    out = {}
    for col in rows[0]:
        if col.startswith("esjd_"):
            total = sum(float(r["proposals"]) * float(r[col]) for r in rows
                        if math.isfinite(float(r[col])))
            metric = col[len("esjd_"):]
            out[f"esjd_{metric}_targeted"] = total / proposals
            out[f"esjdPerSec_{metric}_targeted"] = total / seconds
    return out


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
                              "treeESS_{min,p10,median,max}_{rf,kf}")
    parser.add_argument("--tree-burnin-percent", type=float, default=10.0,
                         help="burn-in percent for --trees, by tree count (matches loganalyser -b)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--rep", required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--targeted-operators", nargs="*", default=[],
                         help="operator ids for the *_targeted columns (datasets.yaml)")
    args = parser.parse_args()

    runtime_sec = read_runtime_seconds(args.runtime)
    runtime_hours = runtime_sec / 3600.0

    opstats = read_tsv(args.opstats)
    chain = chain_row(opstats)

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
    if args.targeted_operators:
        for col, val in targeted_metrics(opstats, set(args.targeted_operators)).items():
            row[col] = fmt(val)

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
        for metric in ("rf", "kf"):
            for stat in ("min", "p10", "median", "max"):
                row[f"treeESS_{stat}_{metric}"] = fmt(tree_ess[metric][stat])
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
        + (f" treeESS_p10_rf={row.get('treeESS_p10_rf')} treeESS_p10_kf={row.get('treeESS_p10_kf')}"
           if args.trees is not None else "")
    )


if __name__ == "__main__":
    main()
