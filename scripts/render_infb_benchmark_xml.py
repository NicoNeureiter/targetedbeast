"""Render a BenchmarkMCMC variant of the infB (141-taxon) template for the Phase 0 step 4
follow-up experiment (infB full / lassaL@100).

infB has no taxon-subsampling and no {target_weight_transform*} slots (checked directly against
`examples/infb_HA_141/template.xml`) - only {chain_length} and {edge_weights}. It's used at its
full 141 taxa, always. Mirrors render_benchmark_xml.py's render_run_xml (same <run>/<state>
rewrite, same RNNIMetric injection pattern) but points the taxonset at infB's own TaxonSet id
(`TaxonSet.infB_HA`) and the ParsimonyWeights2 `data` at infB's own alignment id (`infB_HA`)
instead of lassaL's `@taxonSet` / `@alignment`.
"""
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "examples" / "infb_HA_141" / "template.xml"

EDGE_WEIGHTS_XML = {
    "constant": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
    "parsimony": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ParsimonyWeights2" '
                 'tree="@tree" maxWeight="20" data="@infB_HA"/>',
    "pca": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
           'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
}


def render_run_xml(template_text: str, setting: str, chain_length: int, out_dir: Path,
                    jump_interval: int, pre_burnin: int) -> str:
    xml = template_text.format(
        edge_weights=EDGE_WEIGHTS_XML[setting],
        chain_length=str(chain_length),
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

    state_needle = '<state id="state" spec="State" storeEvery="10000">'
    if state_needle not in xml:
        raise SystemExit("template.xml no longer contains expected <state> opening tag")
    xml = xml.replace(
        state_needle,
        '<metric id="rnni" spec="beastlabs.evolution.tree.RNNIMetric" taxonset="@TaxonSet.infB_HA"/>\n'
        f'        {state_needle}',
    )

    xml = xml.replace('fileName="$(filebase).log"', f'fileName="{out_dir / "run.log"}"')
    xml = xml.replace('fileName="$(filebase).trees"', f'fileName="{out_dir / "run.trees"}"')

    return xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", choices=list(EDGE_WEIGHTS_XML), nargs="+", required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--chain-length", type=int, required=True,
                         help="post-burnin chain length (this is what statistics are computed over)")
    parser.add_argument("--pre-burnin", type=int, default=0)
    parser.add_argument("--jump-interval", type=int, default=10)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    for setting in args.settings:
        for rep in range(args.reps):
            out_dir = args.out_root / setting / f"rep{rep}"
            out_dir.mkdir(parents=True, exist_ok=True)
            xml = render_run_xml(template_text, setting, args.chain_length, out_dir,
                                  args.jump_interval, args.pre_burnin)
            (out_dir / "run.xml").write_text(xml, encoding="utf-8")
            print(f"Wrote {out_dir / 'run.xml'}")


if __name__ == "__main__":
    main()
