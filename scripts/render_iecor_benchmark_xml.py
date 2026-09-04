"""Render a BenchmarkMCMC variant of the iecor (41-taxon) template for the constant vs
PCAWeights2 evaluation.

Modeled directly on `render_infb_benchmark_xml.py` (same <run>/<state>/RNNIMetric rewrite
pattern), but there is no pre-built `template.xml` with `{chain_length}`/`{edge_weights}`
placeholders to fill in for iecor - the two committed files
(`examples/iecor/vanilla/iecor.xml`, `examples/iecor/pca/iecor.xml`) differ from each other
directly in the XML text. `vanilla/iecor.xml` is used as the base (confirmed the simpler of the
two: it has no active edge-weights distribution and no targeted operators wired up at all - its
`<!-- ... -->`-commented `edgeweights` alternatives are all inert and its tree-topology operators
are plain `WilsonBalding`/`Exchange`, not `targetedbeast.operators.*`). That means, unlike infB,
vanilla by itself has no operator that would ever consume an `edgeweights` distribution, so a
"constant vs pca" comparison needs one added: the four targeted operators
(`RangeSlide`/`TargetedWilsonBalding`/`TargetedWilsonBaldingFixedHeight`/`WeightedWideOperator`)
are spliced in verbatim from `pca/iecor.xml` (that file already wires them up, just against its
own `PCAWeights` rather than the `PCAWeights2` convention used elsewhere in this evaluation),
replacing vanilla's `WilsonBalding`/`WideExchange` pair. Every other operator weight (including
`pca/iecor.xml`'s own `SubtreeSlide` weight bump 2.0->5.0) is left as vanilla has it, since that
tuning is orthogonal to the constant/pca comparison this renderer exists for.

Taxon set: iecor has only one `TaxonSet` element (`TaxonSet.five`, built off a site-filtered view
of the `focalsubset` alignment - `FilteredAlignment` filters sites, not taxa, so it carries the
same 41 taxa as the full alignment and is what the tree itself uses), so the RNNIMetric's
taxonset points there. Edge-weights `data` points at `@focalsubset` (iecor's actual alignment id,
not infB's `infB_HA`/lassaL's `@alignment`).
"""
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "examples" / "iecor" / "vanilla" / "iecor.xml"

EDGE_WEIGHTS_XML = {
    "constant": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@Tree.t:tree"/>',
    "pca": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
           'tree="@Tree.t:tree" maxWeight="20" data="@focalsubset" dimension="10"/>',
}

# Verbatim from examples/iecor/pca/iecor.xml - the targeted-operator block that consumes
# "@edgeweights". vanilla has no equivalent (it uses plain WilsonBalding/Exchange instead), so
# this is spliced in, not templated in place.
TARGETED_OPERATORS_XML = (
    '        <operator id="rangeSlide        " spec="targetedbeast.operators.RangeSlide" weight="2" '
    'tree="@Tree.t:tree" edgeWeights="@edgeweights" useWeightedStep="true"/>\n'
    '        <operator id="targetedWilsonBalding" spec="targetedbeast.operators.TargetedWilsonBalding" '
    'tree="@Tree.t:tree" weight="5" edgeWeights="@edgeweights" useEdgeLength="false"/>\n'
    '        <operator id="targetedWilsonBaldingFixedHeight" '
    'spec="targetedbeast.operators.TargetedWilsonBaldingFixedHeight" tree="@Tree.t:tree" weight="15" '
    'edgeWeights="@edgeweights" useEdgeLength="false"/>\n'
    '        <operator id="TargetedWide" spec="targetedbeast.operators.WeightedWideOperator" '
    'tree="@Tree.t:tree" weight="10" edgeWeights="@edgeweights"/>'
)

# vanilla's own (non-targeted) tree-topology operators, removed in favour of the block above.
VANILLA_OPERATORS_XML = (
    '        <operator id="WilsonBalding    " spec="WilsonBalding" tree="@Tree.t:tree" weight="20"/>\n'
    '        <operator id="WideExchange     " spec="Exchange" isNarrow="false" tree="@Tree.t:tree" weight="10"/>'
)

EDGE_WEIGHTS_ANCHOR = '                <!-- Weights for targeted operators -->'


def render_run_xml(template_text: str, setting: str, chain_length: int, out_dir: Path,
                    jump_interval: int, pre_burnin: int) -> str:
    xml = template_text

    if EDGE_WEIGHTS_ANCHOR not in xml:
        raise SystemExit("iecor.xml no longer contains the expected edge-weights comment anchor")
    xml = xml.replace(
        EDGE_WEIGHTS_ANCHOR,
        f'{EDGE_WEIGHTS_ANCHOR}\n                {EDGE_WEIGHTS_XML[setting]}',
    )

    if VANILLA_OPERATORS_XML not in xml:
        raise SystemExit("iecor.xml no longer contains the expected WilsonBalding/WideExchange operators")
    xml = xml.replace(VANILLA_OPERATORS_XML, TARGETED_OPERATORS_XML)

    run_needle = '<run id="mcmc" spec="MCMC" chainLength="10000000">'
    if run_needle not in xml:
        raise SystemExit("iecor.xml no longer contains expected <run> opening tag")
    stats_out = out_dir / "run"
    xml = xml.replace(
        run_needle,
        '<run id="mcmc" spec="targetedbeast.benchmark.BenchmarkMCMC"'
        f' statsOut="{stats_out}" jumpInterval="{jump_interval}"'
        f' preBurnin="{pre_burnin}" chainLength="{chain_length}">',
    )

    state_needle = '<state id="state" spec="State" storeEvery="25000">'
    if state_needle not in xml:
        raise SystemExit("iecor.xml no longer contains expected <state> opening tag")
    xml = xml.replace(
        state_needle,
        '<metric id="rnni" spec="beastlabs.evolution.tree.RNNIMetric" taxonset="@TaxonSet.five"/>\n'
        f'        {state_needle}',
    )

    xml = xml.replace("fileName=\"$(filebase).log\"", f'fileName="{out_dir / "run.log"}"')
    xml = xml.replace("fileName=\"$(filebase).trees\"", f'fileName="{out_dir / "run.trees"}"')

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
