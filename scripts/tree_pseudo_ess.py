"""Pseudo-ESS for a BEAST .trees file, computed via TreeTracer's own machinery
(Lanfear et al. 2016: RF distance from every sampled tree to ~100 random
reference trees, univariate ESS on each resulting trace, report min/median/max).

Reuses TreeTracer's library code directly (rapidtrees for pairwise RF,
treetracer.ess.pseudo_ess for the ESS estimator) rather than TreeTracer's own
Dash app / subprocess-worker plumbing, none of which this needs. Must be run
with the interpreter from TreeTracer's own venv (rapidtrees is a compiled
extension installed there, not in this project's venv):

    /Users/nneureiter/Projects/treetracer/.venv/bin/python scripts/tree_pseudo_ess.py \\
        run.trees --burnin-percent 10

Prints one JSON line to stdout: {"n_trees", "n_trees_used", "min", "median", "max", "n_refs_used"}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TREETRACER_ROOT = Path("/Users/nneureiter/Projects/treetracer/src")

TRANSLATE_LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?),?\s*$")
TREE_LINE_RE = re.compile(r"^\s*tree\s+(\S+)\s*=\s*(?:\[&[A-Z]+\]\s*)?(\(.*;)\s*$", re.IGNORECASE)


def parse_nexus_trees(path: Path) -> tuple[dict[str, str], list[str], list[str]]:
    """Return (translate_map, tree_names, newicks) from a BEAST .trees file."""
    translate_map: dict[str, str] = {}
    names: list[str] = []
    newicks: list[str] = []

    in_translate = False
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not in_translate and stripped.lower() == "translate":
                in_translate = True
                continue
            if in_translate:
                if stripped == ";":
                    in_translate = False
                    continue
                m = TRANSLATE_LINE_RE.match(line.rstrip("\n"))
                if m:
                    num, taxon = m.groups()
                    translate_map[num] = taxon.rstrip(",")
                continue
            m = TREE_LINE_RE.match(line)
            if m:
                tree_name, newick = m.groups()
                names.append(tree_name)
                newicks.append(newick)

    return translate_map, names, newicks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trees_file", type=Path)
    parser.add_argument("--burnin-percent", type=float, default=10.0,
                         help="Percent of trees (by count, in file order) to discard as burn-in. Default 10.")
    parser.add_argument("--n-refs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unrooted", action="store_true",
                         help="Compare bipartitions instead of rooted clades. Default: rooted (BEAST convention).")
    args = parser.parse_args()

    sys.path.insert(0, str(TREETRACER_ROOT))
    import numpy as np
    import rapidtrees
    from treetracer.ess.pseudo_ess import compute_pseudo_ess

    translate_map, names, newicks = parse_nexus_trees(args.trees_file)
    n_trees = len(newicks)
    if n_trees == 0:
        raise SystemExit(f"No trees found in {args.trees_file}")

    burnin = int(n_trees * args.burnin_percent / 100.0)
    names = names[burnin:]
    newicks = newicks[burnin:]
    n_trees_used = len(newicks)

    map_list = [translate_map] if translate_map else []
    map_indices = [0] * n_trees_used if translate_map else None

    result_names, rf_bytes, _leaf_names, _n_bip, _presence_bytes, _bip_bytes = (
        rapidtrees.pairwise_rf_with_snapshots_from_newick_iter(
            names, iter(newicks), map_list, map_indices,
            rooted=not args.unrooted, progress=None,
        )
    )
    n = len(result_names)
    rf_matrix = np.frombuffer(rf_bytes, dtype=np.uint32).reshape(n, n).copy()

    result = compute_pseudo_ess(rf_matrix, n_refs=args.n_refs, seed=args.seed)
    print(json.dumps({
        "n_trees": n_trees,
        "n_trees_used": n_trees_used,
        "min": result["min"],
        "median": result["median"],
        "max": result["max"],
        "n_refs_used": result["n_refs_used"],
    }))


if __name__ == "__main__":
    main()
