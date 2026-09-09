"""Render a BenchmarkMCMC variant of the infB (141-taxon) template for the Phase 0 step 4
follow-up experiment (infB full / lassaL@100), and the weight-transform baseline that extends it.

infB has no taxon-subsampling. It's used at its full 141 taxa, always. Mirrors
render_benchmark_xml.py's render_run_xml (same <run>/<state> rewrite, same RNNIMetric injection
pattern) but points the taxonset at infB's own TaxonSet id (`TaxonSet.infB_HA`) and the
ParsimonyWeights2/PCAWeights2 `data` at infB's own alignment id (`infB_HA`) instead of lassaL's
`@taxonSet` / `@alignment`. `examples/infb_HA_141/template.xml` carries the same
`{target_weight_transform}` / `{target_weight_transform_ref}` / `{target_weight_transform_log}`
slots as `examples/lassaL_kli22/template.xml`, added for the "pca_grid" setting below.
"""
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "examples" / "infb_HA_141" / "template.xml"

# Training data for the "learned" setting: reuses the harness's own completed `constant` cells for
# the same (dataset, tier) - see _learned_edge_weights_xml() below, which discovers them from
# out_dir at render time (posterior samples are exchangeable regardless of which setting produced
# them, so there is no reason to pin this to one historical run - see render_run_xml's docstring).

EDGE_WEIGHTS_XML = {
    "constant": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
    # Built dynamically per out_dir by _learned_edge_weights_xml() - see render_run_xml.
    "learned": None,
    "parsimony": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ParsimonyWeights2" '
                 'tree="@tree" maxWeight="20" data="@infB_HA"/>',
    "pca": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
           'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Same PCAWeights2 as "pca", with a GridTuner-adaptive PowerOffsetTransform layered on top
    # (see WEIGHT_TRANSFORM_XML below) - the question this setting exists to answer is whether
    # the transform improves on PCAWeights2 alone, so the edge weights themselves are identical.
    "pca_grid": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Same PCAWeights2 again, paired with the "grid_low" transform below (see GRID_EXPONENTS_LOW).
    "pca_grid_low": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                     'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Same PCAWeights2 again, paired with the "grid_mid" transform below (see GRID_EXPONENTS_MID).
    "pca_grid_mid": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                     'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Reference point: PCAWeights2 wired to a PowerOffsetTransform pinned at exponent=2/offset=0
    # (the operators' own hardcoded default) with optimise="false" - must be statistically
    # indistinguishable from plain "pca" if the transform/rendering pipeline is correct.
    "pca_fixed2": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                  'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Same PCAWeights2/grid as "pca_grid_mid", but GridTuner's reward is squared RNNI-distance
    # jump (TreeJumpFunction) instead of squared likelihood jump - see "grid_mid_rnni" below.
    "pca_grid_mid_rnni": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                          'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Same PCAWeights2 as "pca_grid_mid", but with a 0.1-uniform-mixture offset floor on both
    # transforms and edgeWeightTransform independently adapted from targetWeightTransform instead
    # of left untransformed - see "grid_mid_dual" below.
    "pca_grid_mid_dual": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                          'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Same PCAWeights2 as "pca_grid_mid", but targetWeightTransform is a RatioGridTuner coupled to
    # edgeWeightTransform's live exponent instead of an independent GridTuner, and the offset floor
    # is lowered to OFFSET_MIXTURE_0_05 - see "grid_mid_ratio" below.
    "pca_grid_mid_ratio": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                           'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
}

# Grid points and burnin below mirror scripts/generate_xmls.py's _grid_transform (the lassaL
# renderer), the only other place this pattern is proven, with one deliberate deviation: lassaL
# has no separate pre-burnin phase, so its GRID_BURNIN=10,000 has to absorb the chain's entire
# transient by itself. Here, core MCMC's preBurnin (100,000 at the thorough tier) already runs
# to completion before operator.optimize() is ever called (MCMC.java: optimize() is gated on
# `sampleNr >= 0`, i.e. strictly post-preBurnin) - and this project's own infB convergence check
# ("infB - 5x extended run" in docs/evaluation/where-targeting-wins.md) already confirmed the
# posterior trace is flat within the first few hundred post-preBurnin samples at that budget. So
# the tuner does not need to relearn to discard a steep transient; a modest safety margin covers
# residual noise right after preBurnin ends.
GRID_EXPONENTS = [8, 12, 16, 24]
GRID_BURNIN = 5_000

# Follow-up to GRID_EXPONENTS: that grid (8/12/16/24) only ever explored exponents pointier than
# the operators' own hardcoded default (2.0, plain squaring) - it could never let GridTuner settle
# back at, or flatter than, the default. This grid spans flatter-than-default (0.5), raw/neutral
# (1), the historical default (2), and moderately pointier (4), so the tuner can actually choose
# among "flatten", "leave alone", and "sharpen" rather than only degrees of "sharpen".
GRID_EXPONENTS_LOW = [0.5, 1, 2, 4]
# maxExponent must be >= the grid's largest value (4); matching it exactly (rather than adding
# headroom, as e.g. maxExponent=8 would) mirrors the pattern the original "grid" setting used
# (maxExponent == max(GRID_EXPONENTS)) and keeps the transform's hard cap tied to the range this
# grid is actually designed to explore.
GRID_MAX_EXPONENT_LOW = max(GRID_EXPONENTS_LOW)

# pca_grid settled at its floor (8.0) in all 4 reps; pca_grid_low settled at its ceiling (4.0) in
# all 4 reps - a boundary solution on both sides of the gap between the two grids. This grid
# spans that gap directly, including the historical default (2) and the pca_grid_low ceiling (4)
# as internal points plus one step past it (6), so a leader at 4 or 6 (not pinned to either edge)
# would confirm the true optimum sits inside [2, 6] rather than being pushed there by grid geometry.
GRID_EXPONENTS_MID = [2, 3, 4, 6]
GRID_MAX_EXPONENT_MID = max(GRID_EXPONENTS_MID)

# WeightTransform.apply() mixes in a uniform distribution with weight a/(1+a), a being the
# `offset` coefficient (see WeightTransform's Input doc) - solving a/(1+a) = 0.1 for a. Applied as
# a fixed, non-gridded floor (GridTuner only ever tunes the exponent - see its class doc) so every
# edge/candidate always keeps a nonzero share of proposal mass, without tuning a second
# coefficient: getting clean signal out of one gridded exponent was hard enough at this chain
# length, so the offset is fixed rather than added as a second grid dimension.
OFFSET_MIXTURE_0_1 = 1.0 / 9.0

# Same mixture-weight solve as OFFSET_MIXTURE_0_1, but for a 0.05 floor instead of 0.1: solving
# a/(1+a) = 0.05 for a. Used by "grid_mid_ratio" below; does not replace OFFSET_MIXTURE_0_1, which
# stays wired to the already-published "grid_mid_dual" setting.
OFFSET_MIXTURE_0_05 = 1.0 / 19.0

WEIGHT_TRANSFORM_XML = {
    "none": ("", "", ""),
    # A shared PowerOffsetTransform whose exponent is chosen by GridTuner from GRID_EXPONENTS.
    # explorationProb=0.1 keeps the choice adaptive rather than a uniform sweep. Reward is the
    # squared change in likelihood per proposal (ESJD) - acceptance alone drives the exponent
    # straight to its ceiling, per docs/WeightTransformAdaptation.md.
    "grid": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="0.0" maxExponent="{max(GRID_EXPONENTS)}.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward idref="likelihood"/>\n'
        '            </tuner>\n'
        '        </targetWeightTransform>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    # Same wiring as "grid", but over GRID_EXPONENTS_LOW instead.
    "grid_low": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="0.0" maxExponent="{GRID_MAX_EXPONENT_LOW}.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS_LOW)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward idref="likelihood"/>\n'
        '            </tuner>\n'
        '        </targetWeightTransform>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    # Same wiring again, over GRID_EXPONENTS_MID.
    "grid_mid": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="0.0" maxExponent="{GRID_MAX_EXPONENT_MID}.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS_MID)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward idref="likelihood"/>\n'
        '            </tuner>\n'
        '        </targetWeightTransform>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    # Same grid/burnin/explorationProb as "grid_mid", but the reward is squared RNNI-distance jump
    # (TreeJumpFunction, targetedbeast.weighttransform) rather than squared likelihood jump. Reuses
    # the "rnni" RNNIMetric instance render_run_xml() already declares before <state> for jump
    # logging - render_run_xml() inserts that <metric id="rnni".../> ahead of this element in
    # document order, so the idref resolves.
    "grid_mid_rnni": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="0.0" maxExponent="{GRID_MAX_EXPONENT_MID}.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS_MID)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward spec="targetedbeast.weighttransform.TreeJumpFunction"'
        ' tree="@tree" treeMetric="@rnni"/>\n'
        '            </tuner>\n'
        '        </targetWeightTransform>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    # Same grid/burnin/explorationProb/reward as "grid_mid", but with the OFFSET_MIXTURE_0_1 floor
    # instead of offset="0.0" - pairs with EDGE_WEIGHT_TRANSFORM_XML's "grid_mid" entry below,
    # which gives edgeWeightTransform its own independently-adapted exponent instead of leaving
    # source-edge weights untransformed.
    "grid_mid_dual": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="{OFFSET_MIXTURE_0_1}" maxExponent="{GRID_MAX_EXPONENT_MID}.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS_MID)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward idref="likelihood"/>\n'
        '            </tuner>\n'
        '        </targetWeightTransform>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    # Regularized coupling: targetWeightTransform's exponent is a bounded ratio (0.7/1.0/1.4) on
    # edgeWeightTransform's own live, per-proposal exponent (RatioGridTuner), instead of an
    # independent GridTuner over its own absolute grid (see "grid_mid_dual" above). maxExponent=9.0
    # covers the worst case: edge's grid max (6.0) x ratio grid max (1.4) = 8.4. Pairs with
    # EDGE_WEIGHT_TRANSFORM_XML's "grid_mid_ratio" entry below, which the reference="@edgeWeight
    # Transform" idref requires to be declared first in document order - see template.xml's
    # {edge_weight_transform}/{target_weight_transform} slot order inside targetedWilsonBalding.
    "grid_mid_ratio": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="{OFFSET_MIXTURE_0_05}" maxExponent="9.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.RatioGridTuner"'
        ' reference="@edgeWeightTransform" ratios="0.7 1.0 1.4"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward idref="likelihood"/>\n'
        '            </tuner>\n'
        '        </targetWeightTransform>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    # Reference point (see "pca_fixed2" above): no tuner, pinned at the operators' own hardcoded
    # default. optimise="false" means startProposal()/optimize() are no-ops - the transform is
    # mathematically w^2 with zero offset on every proposal, identical to the no-transform path.
    "fixed2": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        ' exponent="2.0" offset="0.0" optimise="false"/>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
}

# Which transform variant each edge-weights setting is rendered with.
WEIGHT_TRANSFORM_FOR_SETTING = {
    "constant": "none",
    "parsimony": "none",
    "pca": "none",
    "learned": "none",
    "pca_grid": "grid",
    "pca_grid_low": "grid_low",
    "pca_grid_mid": "grid_mid",
    "pca_fixed2": "fixed2",
    "pca_grid_mid_rnni": "grid_mid_rnni",
    "pca_grid_mid_dual": "grid_mid_dual",
    "pca_grid_mid_ratio": "grid_mid_ratio",
}

# edgeWeightTransform (source-edge weights, see targetedbeast.operators.TargetedWeightOperator) is
# a separate WeightTransform Input from targetWeightTransform on every operator that carries one -
# left unwired ("none") in every setting above, so source-edge selection has always used raw,
# untransformed weights. This "grid_mid" entry gives it its own PowerOffsetTransform/GridTuner,
# independent of (not shared with) targetWeightTransform's - same grid and reward as
# WEIGHT_TRANSFORM_XML's "grid_mid_dual" target transform, but its own id, so the two adapt
# separately and TargetedWeightOperator.optimize()/startTransformProposal() drive each once.
EDGE_WEIGHT_TRANSFORM_XML = {
    "none": ("", "", ""),
    "grid_mid": (
        '<edgeWeightTransform id="edgeWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="{OFFSET_MIXTURE_0_1}" maxExponent="{GRID_MAX_EXPONENT_MID}.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS_MID)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward idref="likelihood"/>\n'
        '            </tuner>\n'
        '        </edgeWeightTransform>',
        ' edgeWeightTransform="@edgeWeightTransform"',
        '<log idref="edgeWeightTransform"/>',
    ),
    # Same as "grid_mid" above except for the offset value (OFFSET_MIXTURE_0_05, not _0_1) - pairs
    # with WEIGHT_TRANSFORM_XML's "grid_mid_ratio" target transform, whose RatioGridTuner reads
    # this transform's live exponent every proposal.
    "grid_mid_ratio": (
        '<edgeWeightTransform id="edgeWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="2.0" offset="{OFFSET_MIXTURE_0_05}" maxExponent="{GRID_MAX_EXPONENT_MID}.0">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS_MID)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        '                <reward idref="likelihood"/>\n'
        '            </tuner>\n'
        '        </edgeWeightTransform>',
        ' edgeWeightTransform="@edgeWeightTransform"',
        '<log idref="edgeWeightTransform"/>',
    ),
}

# Which edge-transform variant each edge-weights setting is rendered with. Defaults to "none"
# (source-edge weights untransformed) unless overridden below.
EDGE_WEIGHT_TRANSFORM_FOR_SETTING = {
    "pca_grid_mid_dual": "grid_mid",
    "pca_grid_mid_ratio": "grid_mid_ratio",
}


def _learned_edge_weights_xml(out_dir: Path) -> str:
    """Build the "learned" setting's LearnedWeights element, training on the sibling `constant`
    cells for the same (dataset, tier) - out_dir is <dataset>/<tier>/learned/rep<rep>, so
    out_dir.parent.parent/constant holds those cells. Posterior samples are exchangeable
    regardless of which setting produced them, so any completed `constant` reps at this tier are
    valid training data - discovered here rather than hardcoded, since reps-per-tier varies.
    """
    constant_dir = out_dir.parent.parent / "constant"
    rep_dirs = sorted(
        (p for p in constant_dir.glob("rep*")
         if (p / "run.trees").exists() and (p / "run.log").exists()),
        key=lambda p: int(p.name[len("rep"):]),
    )
    if not rep_dirs:
        raise SystemExit(
            f'No completed `constant` reps found under "{constant_dir}" (need rep*/run.trees + '
            'rep*/run.log in each). The "learned" setting trains on constant\'s posterior '
            "samples - run constant's cells for this tier first, e.g. via the Snakefile "
            "(`snakemake -j... runs/<dataset>/<tier>/summary.tsv`), then re-render learned."
        )
    tree_files = '\n        '.join(
        f'<trainingTreeFile>{p / "run.trees"}</trainingTreeFile>' for p in rep_dirs
    )
    log_files = '\n        '.join(
        f'<trainingLogFile>{p / "run.log"}</trainingLogFile>' for p in rep_dirs
    )
    # Where the fitted model is cached: LearnedWeights trains once (first XML parse that hits this
    # path missing) and loads the cached fit on every subsequent parse within the same process.
    # Scoped under out_dir itself (each rep's own cell) rather than a shared path, since each rep
    # is rendered/run as its own separate BEAST process - there was never a cross-rep sharing
    # benefit, and a shared path is a write/read race if reps ever run concurrently (-j>1).
    trained_weights_file = out_dir / "learned.weights.tsv"
    return ('<distribution id="edgeweights" spec="targetedbeast.edgeweights.LearnedWeights" '
            'tree="@tree" data="@infB_HA" likelihood="@treeLikelihood" maxWeight="20" '
            'dimension="10" nTrainingTrees="3000" maxIterations="3000" learningRate="0.02" '
            'pcaRegularizationWeight="1e-3" slopeRegularizationWeight="1e-3" miniBatchSize="300" '
            'useLogSlopeParameterization="true" trainingTreeBurninPercentage="0" '
            f'trainingTreeThinInterval="1" trainedWeightsFile="{trained_weights_file}">\n'
            f'        {tree_files}\n'
            f'        {log_files}\n'
            '        </distribution>')


def render_run_xml(template_text: str, setting: str, chain_length: int, out_dir: Path,
                    jump_interval: int, pre_burnin: int) -> str:
    transform_element, transform_ref, transform_log = \
        WEIGHT_TRANSFORM_XML[WEIGHT_TRANSFORM_FOR_SETTING[setting]]
    edge_transform_element, edge_transform_ref, edge_transform_log = \
        EDGE_WEIGHT_TRANSFORM_XML[EDGE_WEIGHT_TRANSFORM_FOR_SETTING.get(setting, "none")]
    edge_weights = (_learned_edge_weights_xml(out_dir) if setting == "learned"
                     else EDGE_WEIGHTS_XML[setting])
    xml = template_text.format(
        edge_weights=edge_weights,
        chain_length=str(chain_length),
        target_weight_transform=transform_element,
        target_weight_transform_ref=transform_ref,
        target_weight_transform_log=transform_log,
        edge_weight_transform=edge_transform_element,
        edge_weight_transform_ref=edge_transform_ref,
        edge_weight_transform_log=edge_transform_log,
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
        '        <metric id="kf" spec="targetedbeast.weighttransform.BranchScoreTreeMetric"/>\n'
        f'        {state_needle}',
    )

    xml = xml.replace('fileName="$(filebase).log"', f'fileName="{out_dir / "run.log"}"')
    xml = xml.replace('fileName="$(filebase).trees"', f'fileName="{out_dir / "run.trees"}"')

    return xml


def main() -> None:
    # Ad hoc single-tree mode, outside the Snakefile grid. out_dir is <out-root>/<setting>/rep<rep>
    # - the same shape render_run_xml expects, so "learned" works here too, but only if the caller
    # has already populated <out-root>/constant/rep*/{run.trees,run.log} themselves first.
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
