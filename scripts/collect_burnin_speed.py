"""Collect one cell's burn-in-speed result (Phase 3c of targetedbeast-evaluation-pipeline.md).

Reads a single `<statsOut>.burnin.tsv` (BenchmarkMCMC's own output, written only when
referenceLowerBound/referenceUpperBound are set) and writes a one-row summary tagged with
dataset/setting/rep/seed, mirroring collect_stats.py's per-cell-then-aggregate shape but for the
burn-in-speed axis instead of opstats/ESS. No statistical analysis here by design - the Snakefile's
own `burnin_speed_summary` rule just concatenates these rows across settings/reps into one table
per dataset; pairing settings by rep and any further analysis happens downstream of the pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def read_burnin_tsv(path: Path) -> dict[str, str]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Unexpected burn-in-speed output in {path}")
    header = lines[0].split("\t")
    values = lines[1].split("\t")
    return dict(zip(header, values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burnin", type=Path, required=True, help="run.burnin.tsv")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--rep", required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    burnin = read_burnin_tsv(args.burnin)
    row = {
        "dataset": args.dataset,
        "setting": args.setting,
        "rep": args.rep,
        # seed = 1000 + rep, the same fixed convention used throughout this pipeline (never
        # hash(setting, rep)) - not read from opstats.tsv to keep this script's only input the
        # burnin.tsv file itself.
        "seed": str(1000 + int(args.rep)),
        **burnin,
    }
    header = list(row.keys())
    values = [row[k] for k in header]
    args.summary_out.write_text("\t".join(header) + "\n" + "\t".join(values) + "\n")

    print(f"{args.dataset}/{args.setting}/rep{args.rep}: reached={row.get('reached')} "
          f"proposalsToTarget={row.get('proposalsToTarget')}")


if __name__ == "__main__":
    main()
