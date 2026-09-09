"""Write cell.manifest.json: per-cell provenance for the evaluation pipeline.

Snakemake's own staleness tracking no longer ties a cell's outputs to Java source mtimes (see
rule run_beast's `mvn -o compile -q &&` shell prefix, Snakefile) - this manifest is the
replacement record of what code and rendered XML actually produced a given cell's outputs, so a
human (or check_reference_freshness.py) can later tell whether a completed reference cell is
still current. Invoked after the BEAST run succeeds (never before) so a failed run leaves no
manifest claiming success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def git_dirty(repo: Path) -> bool:
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    return bool(status.strip())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(repo: Path, xml: Path, dataset: str, tier: str, setting: str, rep: str,
                    seed: int) -> dict:
    return {
        "git_commit": git_commit(repo),
        "git_dirty": git_dirty(repo),
        "dataset": dataset,
        "tier": tier,
        "setting": setting,
        "rep": rep,
        "seed": seed,
        "xml_sha256": sha256_of(xml),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="project root (git repo)")
    parser.add_argument("--xml", type=Path, required=True, help="the cell's rendered run.xml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--rep", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="cell.manifest.json path")
    args = parser.parse_args()

    manifest = build_manifest(args.repo, args.xml, args.dataset, args.tier, args.setting,
                               args.rep, args.seed)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}: commit={manifest['git_commit'][:8]} dirty={manifest['git_dirty']}")


if __name__ == "__main__":
    main()
