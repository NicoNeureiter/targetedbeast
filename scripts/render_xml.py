"""Single entry point for rendering a BenchmarkMCMC run.xml, dispatching to the per-dataset
renderer named in `datasets.yaml` (targetedbeast-evaluation-pipeline.md, Phase 3).

Does not reimplement any XML surgery: `render_infb_benchmark_xml.py` and
`render_iecor_benchmark_xml.py` already expose a standalone `render_run_xml(template_text,
setting, chain_length, out_dir, jump_interval, pre_burnin)` function (used here as-is, via
importlib, not copied). `primates.xml` has no per-dataset renderer script (there was never a
comparison to drive one - see `render_primates` below) and lassaL's `render_benchmark_xml.py` is
parked with lassaL itself (Phase 3a) and deliberately not wired in here.

Usage:
    python scripts/render_xml.py --dataset infb --tier fast --setting constant --rep 0
    python scripts/render_xml.py --dataset primates --tier smoke --setting default --rep 0

Writes runs/<dataset>/<tier>/<setting>/rep<rep>/run.xml and prints its path. Requires PyYAML
(`pip install pyyaml`) - already a Snakemake dependency, so anywhere the Snakefile runs this is
already satisfied.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATASETS_YAML = SCRIPTS_DIR / "datasets.yaml"

# So `importlib.import_module(entry["renderer"])` finds render_infb_benchmark_xml.py etc.
# regardless of the caller's own cwd (Snakemake rules run with cwd unset to this directory).
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_registry(path: Path = DATASETS_YAML) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tier_config(registry: dict, dataset: str, tier: str) -> dict:
    if dataset not in registry:
        raise SystemExit(f"Unknown dataset {dataset!r}; known: {sorted(registry)}")
    entry = registry[dataset]
    if tier not in entry["tiers"]:
        raise SystemExit(
            f"Dataset {dataset!r} has no tier {tier!r}; known tiers: {sorted(entry['tiers'])}"
        )
    return entry["tiers"][tier]


def render_primates(template_text: str, chain_length: int, out_dir: Path, jump_interval: int,
                     pre_burnin: int) -> str:
    """primates.xml has no {slot} template and no per-dataset renderer script: it is small,
    single-setting, and exists only for the smoke tier's crash/sane-acceptance check (Phase 3a),
    so there is nothing to swap and no separate script has ever been needed for it. No
    RNNIMetric is injected - primates.xml declares no TaxonSet, and the smoke tier makes no
    statistical claim, so opstats.tsv gets per-operator accept/reject/timing rows with NaN
    esjd/meanJump columns, same as any other BenchmarkMCMC run with an empty metric list.
    """
    run_needle = '<run spec="MCMC" id="mcmc" chainLength="5000000">'
    if run_needle not in template_text:
        raise SystemExit("primates.xml no longer contains expected <run> opening tag")
    stats_out = out_dir / "run"
    xml = template_text.replace(
        run_needle,
        '<run spec="targetedbeast.benchmark.BenchmarkMCMC" id="mcmc"'
        f' statsOut="{stats_out}" jumpInterval="{jump_interval}"'
        f' preBurnin="{pre_burnin}" chainLength="{chain_length}">',
    )
    xml = xml.replace('fileName="$(filebase).log"', f'fileName="{out_dir / "run.log"}"')
    xml = xml.replace('fileName="$(filebase).trees"', f'fileName="{out_dir / "run.trees"}"')
    return xml


def _inject_reference_band(xml: str, lower: float | None, upper: float | None) -> str:
    """Adds referenceLowerBound/referenceUpperBound attributes to the rendered <run
    spec="targetedbeast.benchmark.BenchmarkMCMC" ...> element, for the burn-in-speed metric
    (Phase 3c). A single injection point here - rather than touching every per-dataset renderer
    script - because every renderer already emits a statsOut attribute on that element (anchor for
    the insertion) and the two are otherwise independent of any dataset-specific XML surgery.
    """
    if lower is None and upper is None:
        return xml
    if lower is None or upper is None:
        raise SystemExit("reference_lower_bound and reference_upper_bound must be given together")
    needle = ' statsOut="'
    index = xml.find(needle)
    if index == -1:
        raise SystemExit("rendered XML has no statsOut attribute to anchor "
                          "referenceLowerBound/referenceUpperBound on")
    insertion = f' referenceLowerBound="{lower}" referenceUpperBound="{upper}"'
    return xml[:index] + insertion + xml[index:]


def _inject_checkpoint_out(xml: str, checkpoint_out: str | None) -> str:
    """Adds a checkpointOut attribute to the rendered BenchmarkMCMC element (Phase 3c, item 1):
    only the dataset's one reference-checkpoint donor run sets this - see rule
    render_reference_checkpoint in the Snakefile. Regular thorough/cross-check cells consume the
    resulting checkpoint file via -resume, not via this attribute."""
    if checkpoint_out is None:
        return xml
    needle = ' statsOut="'
    index = xml.find(needle)
    if index == -1:
        raise SystemExit("rendered XML has no statsOut attribute to anchor checkpointOut on")
    insertion = f' checkpointOut="{checkpoint_out}"'
    return xml[:index] + insertion + xml[index:]


def render(dataset: str, tier: str, setting: str, rep: int, out_dir: Path,
           registry: dict | None = None, reference_lower_bound: float | None = None,
           reference_upper_bound: float | None = None, checkpoint_out: str | None = None) -> Path:
    """Render one (dataset, tier, setting, rep) cell's run.xml into out_dir. Returns its path.

    reference_lower_bound/reference_upper_bound (Phase 3c, burn-in-speed): when both given, wires
    them onto the rendered BenchmarkMCMC element so BurnInSpeedTracker is enabled for this cell.

    checkpoint_out (Phase 3c, shared checkpoint): when given, wires it onto the rendered
    BenchmarkMCMC element as checkpointOut - used only by the dataset's reference-checkpoint donor
    run (see Snakefile), never by a regular tier cell.
    """
    registry = registry or load_registry()
    entry = registry[dataset]
    known = entry["settings"] + entry.get("extra_settings", [])
    if setting not in known:
        raise SystemExit(f"Dataset {dataset!r} has no setting {setting!r}; known: {known}")
    # "burnin-speed" is not a tier: its chain shape is the dataset's burnin_speed entry.
    cfg = entry["burnin_speed"] if tier == "burnin-speed" else tier_config(registry, dataset, tier)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_path = PROJECT_ROOT / entry["template"]
    template_text = template_path.read_text(encoding="utf-8")

    if entry["renderer"] is None:
        if dataset != "primates":
            raise SystemExit(f"Dataset {dataset!r} declares no renderer but isn't primates")
        xml = render_primates(template_text, cfg["chain_length"], out_dir, cfg["jump_interval"],
                               cfg["pre_burnin"])
    else:
        module = importlib.import_module(entry["renderer"])
        if hasattr(module, "load_template"):
            template_text = module.load_template(setting)
        xml = module.render_run_xml(template_text, setting, cfg["chain_length"], out_dir,
                                     cfg["jump_interval"], cfg["pre_burnin"])

    xml = _inject_reference_band(xml, reference_lower_bound, reference_upper_bound)
    xml = _inject_checkpoint_out(xml, checkpoint_out)

    out_path = out_dir / "run.xml"
    out_path.write_text(xml, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out-dir", type=Path,
                         help="defaults to runs/<dataset>/<tier>/<setting>/rep<rep>/")
    parser.add_argument("--reference-band", type=Path, default=None,
                         help="path to a reference_band.json (compute_reference_band.py's output) "
                              "whose lower/upper are wired onto the rendered BenchmarkMCMC element "
                              "as referenceLowerBound/referenceUpperBound (Phase 3c burn-in-speed)")
    parser.add_argument("--checkpoint-out", type=str, default=None,
                         help="path to write the shared checkpoint to (Phase 3c, item 1) - only "
                              "the dataset's reference-checkpoint donor run sets this")
    args = parser.parse_args()

    out_dir = args.out_dir or (
        PROJECT_ROOT / "runs" / args.dataset / args.tier / args.setting / f"rep{args.rep}"
    )
    lower = upper = None
    if args.reference_band is not None:
        band = json.loads(args.reference_band.read_text(encoding="utf-8"))
        lower, upper = band["lower"], band["upper"]
    path = render(args.dataset, args.tier, args.setting, args.rep, out_dir,
                   reference_lower_bound=lower, reference_upper_bound=upper,
                   checkpoint_out=args.checkpoint_out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
