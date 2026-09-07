"""Render a BenchmarkMCMC variant of the lassaL template for the Phase 0 step 4 experiment.

Reuses generate_xmls.py's subsampled_alignment() (seeded, SUBSAMPLE_SEED=42) so taxon sets are
reproducible, but writes a self-contained run.xml per (size, setting, rep) with:
  - <run spec="MCMC" ...> rewritten to spec="targetedbeast.benchmark.BenchmarkMCMC"
  - statsOut/jumpInterval/metric wired to RNNIMetric via the template's existing @taxonSet
  - edge_weights filled with either ConstantWeights or ParsimonyWeights2 (maxWeight=20, matching
    the value already used elsewhere in this corpus for ParsimonyWeights2/PCAWeights2)
  - no weight-transform (target_weight_transform* slots left empty - not part of this comparison)
  - absolute log/trees/alignment paths so each run directory is self-contained
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_xmls as g

EDGE_WEIGHTS_XML = {
    "constant": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
    "parsimony": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ParsimonyWeights2" '
                 'tree="@tree" maxWeight="20" data="@alignment"/>',
    "pca": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
           'tree="@tree" maxWeight="20" data="@alignment" dimension="10"/>',
}


def render_run_xml(template_text: str, setting: str, chain_length: int, alignment_file: Path,
                    out_dir: Path, jump_interval: int, pre_burnin: int) -> str:
    xml = g.render_xml(
        template_text,
        weights_string=EDGE_WEIGHTS_XML[setting],
        transform=("", "", ""),
        chain_length=chain_length,
        alignment_file=alignment_file,
    )

    run_needle = '<run id="mcmc" spec="MCMC" chainLength='
    if run_needle not in xml:
        raise SystemExit("template.xml no longer contains expected <run> opening tag")
    stats_out = out_dir / "run"
    xml = xml.replace(
        run_needle,
        '<run id="mcmc" spec="targetedbeast.benchmark.BenchmarkMCMC"'
        f' statsOut="{stats_out}" jumpInterval="{jump_interval}"'
        f' preBurnin="{pre_burnin}"'
        ' chainLength=',
    )

    # RNNIMetric goes right after <state ...> opens, referencing the template's existing taxonSet.
    state_needle = '<state id="state" spec="State" storeEvery="10000">'
    if state_needle not in xml:
        raise SystemExit("template.xml no longer contains expected <state> opening tag")
    xml = xml.replace(
        state_needle,
        '<metric id="rnni" spec="beastlabs.evolution.tree.RNNIMetric" taxonset="@taxonSet"/>\n'
        f'        {state_needle}',
    )

    # Point loggers at absolute paths inside out_dir instead of relying on cwd/$(filebase).
    xml = xml.replace('fileName="$(filebase).log"', f'fileName="{out_dir / "run.log"}"')
    xml = xml.replace('fileName="$(filebase).trees"', f'fileName="{out_dir / "run.trees"}"')

    return xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--settings", choices=list(EDGE_WEIGHTS_XML), nargs="+", required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--chain-length", type=int, required=True,
                         help="post-burnin chain length (this is what statistics are computed over)")
    parser.add_argument("--pre-burnin", type=int, default=0,
                         help="MCMC preBurnin - discarded before BenchmarkMCMC records anything "
                              "(negative sampleNr, per MCMC.doLoop's -burnIn..chainLength range)")
    parser.add_argument("--jump-interval", type=int, default=10)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    template_text = g.TEMPLATE_PATH.read_text(encoding="utf-8")

    for size in args.sizes:
        alignment_file = g.subsampled_alignment(size)
        for setting in args.settings:
            for rep in range(args.reps):
                out_dir = args.out_root / f"n{size}" / setting / f"rep{rep}"
                out_dir.mkdir(parents=True, exist_ok=True)
                xml = render_run_xml(
                    template_text, setting, args.chain_length, alignment_file, out_dir,
                    args.jump_interval, args.pre_burnin,
                )
                (out_dir / "run.xml").write_text(xml, encoding="utf-8")
                print(f"Wrote {out_dir / 'run.xml'}")


if __name__ == "__main__":
    main()
