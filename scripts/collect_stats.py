"""Collect MCMC run statistics for a single BEAST replicate.

Combines three sources for one `run?.xml` replicate:
  * loganalyser `-oneline` output  -> per-parameter ESS (and other summaries)
  * the `run?.xml.state` file       -> per-operator accept/reject counts
  * a `run?.runtime` file           -> wall-clock runtime in seconds

and writes:
  * a one-row summary TSV: the loganalyser columns extended with
      - lowest ESS across all parameters
      - ESS per hour (lowest ESS / runtime)
      - acceptance rate of the requested operator (TargetedWilsonBalding)
  * a long-format operator TSV: accept/reject/accept-rate for every operator.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def read_runtime_seconds(path: Path) -> float:
	"""Read wall-clock runtime (seconds) from a one-number runtime file."""
	return float(path.read_text().strip())


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


def parse_operators(state_path: Path) -> list[dict]:
	"""Extract the operator stats JSON from the trailing comment of a .state file."""
	text = state_path.read_text()
	match = re.search(r"<!--(.*?)-->", text, re.S)
	if not match:
		raise ValueError(f"No operator JSON comment found in {state_path}")
	data = json.loads(match.group(1))
	return data["operators"]


def accept_rate(op: dict) -> float | None:
	"""Acceptance rate = accept / (accept + reject), or None if no proposals."""
	accept = op.get("accept", 0)
	reject = op.get("reject", 0)
	total = accept + reject
	return accept / total if total > 0 else None


def fmt(x: float | None) -> str:
	return "NA" if x is None else f"{x:.6g}"


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--ess-table", type=Path, required=True, help="loganalyser -oneline output")
	parser.add_argument("--state", type=Path, required=True, help="run?.xml.state file")
	parser.add_argument("--runtime", type=Path, required=True, help="run?.runtime file (seconds)")
	parser.add_argument("--subdir", required=True, help="edge-weight subdirectory name")
	parser.add_argument("--rep", required=True, help="replicate index")
	parser.add_argument(
		"--operator",
		default="targetedWilsonBalding",
		help="operator id whose acceptance rate is added to the summary",
	)
	parser.add_argument("--summary-out", type=Path, required=True)
	parser.add_argument("--operators-out", type=Path, required=True)
	args = parser.parse_args()

	runtime_sec = read_runtime_seconds(args.runtime)
	runtime_hours = runtime_sec / 3600.0

	log_row = read_loganalyser_oneline(args.ess_table)
	low_param, low_ess = lowest_ess(log_row)
	ess_per_hour = low_ess / runtime_hours if runtime_hours > 0 else float("nan")

	operators = parse_operators(args.state)

	# Acceptance rate of the requested operator (e.g. TargetedWilsonBalding).
	op_rate = None
	for op in operators:
		if op.get("id") == args.operator:
			op_rate = accept_rate(op)
			break

	# --- summary row: loganalyser columns extended with our extra metrics ---
	extra = {
		"subdir": args.subdir,
		"rep": args.rep,
		"runtime_sec": fmt(runtime_sec),
		"runtime_hours": fmt(runtime_hours),
		"lowest_ESS": fmt(low_ess),
		"lowest_ESS_param": low_param,
		"ESS_per_hour": fmt(ess_per_hour),
		f"{args.operator}_accept_rate": fmt(op_rate),
	}
	header = list(extra.keys()) + list(log_row.keys())
	values = list(extra.values()) + list(log_row.values())
	args.summary_out.write_text("\t".join(header) + "\n" + "\t".join(values) + "\n")

	# --- long-format operator acceptance table ---
	op_header = ["subdir", "rep", "operator", "accept", "reject", "n_proposals", "accept_rate"]
	op_lines = ["\t".join(op_header)]
	for op in operators:
		accept = op.get("accept", 0)
		reject = op.get("reject", 0)
		op_lines.append(
			"\t".join(
				[
					args.subdir,
					args.rep,
					str(op.get("id", "")),
					str(accept),
					str(reject),
					str(accept + reject),
					fmt(accept_rate(op)),
				]
			)
		)
	args.operators_out.write_text("\n".join(op_lines) + "\n")

	print(
		f"{args.subdir}/run{args.rep}: runtime={runtime_sec:.0f}s "
		f"lowest ESS={low_ess:.1f} ({low_param}) "
		f"ESS/h={ess_per_hour:.1f} {args.operator} accept={fmt(op_rate)}"
	)


if __name__ == "__main__":
	main()
