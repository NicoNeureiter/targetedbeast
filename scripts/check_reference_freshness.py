"""Warn-only report: compare each cell's cell.manifest.json against the current repo state.

Never exits non-zero and never blocks anything - purely informational (see write_cell_manifest.py
for what a manifest records and why Snakemake's own staleness tracking no longer does this job).

Usage:
    python scripts/check_reference_freshness.py runs/infb/thorough/*/rep*/
    python scripts/check_reference_freshness.py --repo /path/to/targetedbeast runs/infb/fast/*/rep*/
"""
from __future__ import annotations

import argparse
import json
import subprocess
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


def status_for(manifest: dict | None, head: str, dirty_now: bool) -> str:
    if manifest is None:
        return "no manifest"
    if manifest.get("git_dirty"):
        return "DIRTY-AT-RUN-TIME"
    if manifest.get("git_commit") == head:
        return "OK"
    return "STALE"


def load_manifest(cell_dir: Path) -> dict | None:
    manifest_path = cell_dir / "cell.manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_dirs", nargs="+", type=Path,
                         help="cell directories to check (each expected to contain "
                              "cell.manifest.json); shell-glob these, e.g. "
                              "runs/infb/thorough/*/rep*/")
    parser.add_argument("--repo", type=Path,
                         default=Path(__file__).resolve().parent.parent,
                         help="project root (git repo); defaults to this script's own repo")
    args = parser.parse_args()

    head = git_commit(args.repo)
    dirty_now = git_dirty(args.repo)

    rows = []
    for cell_dir in sorted(args.cell_dirs):
        manifest = load_manifest(cell_dir)
        status = status_for(manifest, head, dirty_now)
        rows.append((
            manifest.get("dataset", "?") if manifest else "?",
            manifest.get("tier", "?") if manifest else "?",
            manifest.get("setting", "?") if manifest else "?",
            str(manifest.get("rep", "?")) if manifest else "?",
            manifest.get("git_commit", "")[:8] if manifest else "-",
            status,
            str(cell_dir),
        ))

    header = ("dataset", "tier", "setting", "rep", "commit", "status", "cell_dir")
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]

    def fmt_row(r):
        return "  ".join(cell.ljust(w) for cell, w in zip(r, widths))

    print(f"HEAD={head[:8]} dirty={dirty_now}")
    print(fmt_row(header))
    for r in rows:
        print(fmt_row(r))


if __name__ == "__main__":
    main()
