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

# Settings whose targeted operators differ structurally from template.xml's. Such a template may
# differ from template.xml only between the targeted-operators markers - load_template enforces it.
TEMPLATE_FOR_SETTING = {
    "pca_aos_mid": TEMPLATE_PATH.with_name("template_aos.xml"),
}
OPERATORS_BEGIN = "<!-- targeted-operators:begin -->"
OPERATORS_END = "<!-- targeted-operators:end -->"


def _outside_operators(text: str, path: Path) -> tuple[str, str]:
    begin, end = text.find(OPERATORS_BEGIN), text.find(OPERATORS_END)
    if begin == -1 or end < begin:
        raise SystemExit(f"{path} lacks the {OPERATORS_BEGIN} ... {OPERATORS_END} markers")
    return text[:begin], text[end:]


def load_template(setting: str) -> str:
    """The template text for one setting, refusing a variant template that has drifted from
    template.xml outside the targeted-operators block."""
    base = TEMPLATE_PATH.read_text(encoding="utf-8")
    path = TEMPLATE_FOR_SETTING.get(setting)
    if path is None:
        return base
    variant = path.read_text(encoding="utf-8")
    if _outside_operators(variant, path) != _outside_operators(base, TEMPLATE_PATH):
        raise SystemExit(f"{path.name} differs from {TEMPLATE_PATH.name} outside the targeted "
                         "operators; re-copy the shared parts")
    return variant


# Pinned target-weight exponents, by setting-name suffix ("1p5" -> pca_fixed1p5).
FIXED_EXPONENTS = {"0p1": 0.1, "0p2": 0.2, "0p5": 0.5, "1": 1.0, "1p5": 1.5, "2": 2.0, "3": 3.0,
                   "4": 4.0, "6": 6.0, "8": 8.0}
PCA_FIXED_EXPONENTS = ("1", "1p5", "3", "4", "6", "8")

# PartialWeights returns a softmax over candidate log-likelihoods, so a pinned exponent b acts as an
# inverse temperature: softmax(b * loglik). It is already sharply peaked, so the sweep flattens
# (b < 1); 0.1 is WeightTransform's lower bound on the exponent.
PARTIAL_FIXED_EXPONENTS = ("0p1", "0p2", "0p5", "1")
EDGE_WEIGHTS_PARTIAL = ('<distribution id="edgeweights" spec="targetedbeast.edgeweights.PartialWeights" '
                        'tree="@tree" maxWeight="20" data="@infB_HA" likelihood="@treeLikelihood"/>')
# PartialWeights per ThreadedTreeLikelihood slice, scored in parallel; same weights as
# EDGE_WEIGHTS_PARTIAL. Threads come from BEAST's -threads flag (threaded_settings in datasets.yaml).
PARTIAL_THREADED_EXPONENTS = ("0p2", "0p5")
EDGE_WEIGHTS_PARTIAL_THREADED = ('<distribution id="edgeweights" '
                                 'spec="targetedbeast.edgeweights.ThreadedPartialWeights" '
                                 'tree="@tree" maxWeight="20" likelihood="@treeLikelihood"/>')

# Source edges from one weight class, targets from ThreadedPartialWeights at b = 0.5; the baseline,
# partial_thr_fixed0p5, takes its source weights from PartialWeights itself (branch lengths).
SOURCE_WEIGHTS_XML = {
    "uniform": '<distribution id="sourceWeights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
    "pca": ('<distribution id="sourceWeights" spec="targetedbeast.edgeweights.PCAWeights2" '
            'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>'),
    "length": '<distribution id="sourceWeights" spec="targetedbeast.edgeweights.BranchLengthWeights" tree="@tree"/>',
}


# Source weights mixed towards branch lengths through the edge transform's offset: offset a gives
# weight a/(1+a) to the normalised branch lengths. 0 reproduces the unmixed weights through the same
# code path, 1 is an even mix, 100 (the offset's upper bound) is 99% branch length.
LENGTH_MIX_OFFSETS = {"0": "0.0", "0p5": "1.0", "len": "100.0"}


TARGET_WEIGHTS_XML = {
    "partial": ('<distribution id="targetWeights" spec="targetedbeast.edgeweights.ThreadedPartialWeights" '
                'tree="@tree" maxWeight="20" likelihood="@treeLikelihood"/>'),
    "pca": ('<distribution id="targetWeights" spec="targetedbeast.edgeweights.PCAWeights2" '
            'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>'),
}


def edge_weights_source_target(source: str, target: str = "partial") -> str:
    return "\n".join([
        SOURCE_WEIGHTS_XML[source],
        TARGET_WEIGHTS_XML[target],
        '<distribution id="edgeweights" spec="targetedbeast.edgeweights.SourceTargetWeights" '
        'source="@sourceWeights" target="@targetWeights"/>',
    ])


# Source weights length^c: BranchLengthWeights through the edge transform's exponent c.
LENGTH_EXPONENTS = {"0p5": "0.5", "1": "1.0", "1p5": "1.5", "2": "2.0"}

# Uniform floor on the topology operators' target weights at exponent 3: offset a gives the uniform
# weight a/(1+a). With the split source exponents and the slide on its own transforms.
TARGET_OFFSETS = {"0": "0.0", "0p05": "0.05", "0p2": "0.2"}
TARGET_OFFSET_SETTINGS = tuple(f"pca_split_toff{k}" for k in TARGET_OFFSETS)


# treeLikelihood's spec, by setting. PartialWeights needs TreeLikelihood.getLikelihoodCore(), which
# ThreadedTreeLikelihood does not have; pca_nt is pca on the same likelihood, for comparable timing.
DEFAULT_LIKELIHOOD_SPEC = "beast.base.spec.evolution.likelihood.ThreadedTreeLikelihood"
NON_THREADED_LIKELIHOOD_SPEC = "beast.base.spec.evolution.likelihood.TreeLikelihood"

EDGE_WEIGHTS_PCA = ('<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                    'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>')

EDGE_WEIGHTS_PCA_L2 = EDGE_WEIGHTS_PCA.replace('/>', ' useOneNorm="false"/>')
L2_EXPONENTS = ("1", "1p5", "3")

EDGE_WEIGHTS_XML = {
    "constant": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
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
    # Same PCAWeights2; pca_grid_mid's exponents chosen by AdaptableOperatorSampler instead of
    # GridTuner - the operators live in TEMPLATE_FOR_SETTING's template_aos.xml.
    "pca_aos_mid": '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" '
                    'tree="@tree" maxWeight="20" data="@infB_HA" dimension="10"/>',
    # Same PCAWeights2 with the exponent pinned (fixed2's wiring): where is the optimum, and does
    # GridTuner beat simply fixing it there?
    **{f"pca_fixed{k}": EDGE_WEIGHTS_PCA for k in PCA_FIXED_EXPONENTS},
    "pca_nt": EDGE_WEIGHTS_PCA,
    "partial": EDGE_WEIGHTS_PARTIAL,
    **{f"partial_fixed{k}": EDGE_WEIGHTS_PARTIAL for k in PARTIAL_FIXED_EXPONENTS},
    # Threaded Java likelihood: pca_thr is pca's baseline for the partial_thr settings.
    "pca_thr": EDGE_WEIGHTS_PCA,
    **{f"partial_thr_fixed{k}": EDGE_WEIGHTS_PARTIAL_THREADED for k in PARTIAL_THREADED_EXPONENTS},
    **{f"partial_thr_src_{src}": edge_weights_source_target(src) for src in ("uniform", "pca")},
    **{f"pca_srclen{k}": edge_weights_source_target("length", "pca") for k in LENGTH_EXPONENTS},
    **{f"partial_thr_srclen{k}": edge_weights_source_target("length") for k in LENGTH_EXPONENTS},
    "partial_thr_srclen_grid": edge_weights_source_target("length"),
    "partial_thr_srclen_grid_kf": edge_weights_source_target("length"),
    **{k: edge_weights_source_target("length", "pca") for k in ("pca_split_range12", "pca_split_slide", "pca_split_local", "pca_split_slide_tgt0p5", "pca_split_slide_tgt1", "pca_split_slide_tgt2", "pca_split_grid", "pca_split_grid_wide")},
    **{f"pca_split_toff{k}": edge_weights_source_target("length", "pca") for k in TARGET_OFFSETS},
    # PCA source weights mixed towards branch lengths (EDGE_WEIGHT_TRANSFORM_FOR_SETTING); with
    # PartialWeights targets, mix 0 is partial_thr_src_pca.
    **{f"pca_srcmix{k}": EDGE_WEIGHTS_PCA for k in LENGTH_MIX_OFFSETS},
    **{f"partial_thr_srcmix{k}": edge_weights_source_target("pca") for k in LENGTH_MIX_OFFSETS if k != "0"},
    # PCAWeights2 on Euclidean rather than 1-norm distance in PCA space, at the default exponent
    # and pinned at L2_EXPONENTS.
    "pca_l2": EDGE_WEIGHTS_PCA_L2,
    **{f"pca_l2_fixed{k}": EDGE_WEIGHTS_PCA_L2 for k in L2_EXPONENTS},
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

# Grids for source weights length^c and PartialWeights' target exponent b, tuned together
# (partial_thr_srclen_grid). c = 2 collapsed topology mixing with pca targets; b = 0.2-0.5 was best
# in the fixed PartialWeights sweep. The source side keeps a fixed offset so that zero-length edges
# keep a positive weight.
GRID_EXPONENTS_LENGTH = [0.25, 0.5, 0.75, 1.0, 1.5]
GRID_EXPONENTS_PARTIAL = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
LENGTH_GRID_OFFSET = 0.03

# GridTuner rewards: squared likelihood jump, or squared KF (branch score) distance jump across a
# proposal. Each tuner needs its own TreeJumpFunction, which pairs its before/after calls.
GRID_REWARDS = {
    "likelihood": '<reward idref="likelihood"/>',
    "kf": ('<reward spec="targetedbeast.weighttransform.TreeJumpFunction" tree="@tree">'
           '<treeMetric spec="targetedbeast.weighttransform.KuhnerFelsensteinMetric"/></reward>'),
    "rf": ('<reward spec="targetedbeast.weighttransform.TreeJumpFunction" tree="@tree">'
           '<treeMetric spec="targetedbeast.weighttransform.RobinsonFouldsMetric"/></reward>'),
}


TREE_METRIC_SPEC = {
    "rf": "targetedbeast.weighttransform.RobinsonFouldsMetric",
    "kf": "targetedbeast.weighttransform.KuhnerFelsensteinMetric",
}


def shared_reward(metric: str, id_: str, define: bool) -> str:
    """A tree-jump reward shared by the tuners of one operator group: defined by the first, referenced by the rest."""
    if not define:
        return f'<reward idref="{id_}"/>'
    return (f'<reward id="{id_}" spec="targetedbeast.weighttransform.TreeJumpFunction" tree="@tree">'
            f'<treeMetric spec="{TREE_METRIC_SPEC[metric]}"/></reward>')


def grid_transform(tag: str, id_: str, exponents: list[float], reward: str, offset: float = 0.0) -> str:
    """A grid-tuned PowerOffsetTransform element; tag is edgeWeightTransform or targetWeightTransform.
    reward is a GRID_REWARDS key or a reward element."""
    reward = GRID_REWARDS.get(reward, reward)
    return (
        f'<{tag} id="{id_}" spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="{exponents[len(exponents) // 2]}" offset="{offset}" maxExponent="{max(exponents)}">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(e) for e in exponents)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        f'                {reward}\n'
        '            </tuner>\n'
        f'        </{tag}>'
    )


# Separate tuners for the topology operators and the range slide (pca_split_grid), each on a
# reward matching its job: RF jumps for the topology operators, KF jumps for the slide.
SPLIT_GRID_TOPOLOGY_SOURCE = [0.25, 0.5, 0.75, 1.0]
SPLIT_GRID_TOPOLOGY_TARGET = [1.0, 2.0, 3.0, 4.0, 6.0]
SPLIT_GRID_SLIDE_SOURCE = [0.75, 1.0, 1.5, 2.0, 3.0]
SPLIT_GRID_SLIDE_TARGET = [0.5, 0.75, 1.0, 1.5, 2.0]
# Extended where the tuners ran to an edge: topology c and b downwards, slide c upwards.
SPLIT_GRID_WIDE_TOPOLOGY_SOURCE = [0.1, 0.25, 0.5, 0.75, 1.0]
SPLIT_GRID_WIDE_TOPOLOGY_TARGET = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
SPLIT_GRID_WIDE_SLIDE_SOURCE = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def grid_partial_transform(reward: str) -> tuple[str, str, str]:
    """PartialWeights' target exponent b, grid-tuned over the range the fixed sweep found useful."""
    return (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="0.5" offset="0.0" maxExponent="{max(GRID_EXPONENTS_PARTIAL)}">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS_PARTIAL)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        f'                {GRID_REWARDS[reward]}\n'
        '            </tuner>\n'
        '        </targetWeightTransform>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    )

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
    # Topology operators' target exponent pinned at 3 with a uniform floor (pca_split_toff*).
    **{f"fixed3_off{k}": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="3.0" offset="{a}" optimise="false"/>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ) for k, a in TARGET_OFFSETS.items()},
    "grid_split_topology_wide": (
        grid_transform("targetWeightTransform", "targetWeightTransform", SPLIT_GRID_WIDE_TOPOLOGY_TARGET,
                       shared_reward("rf", "topologyJump", define=False)),
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    "grid_split_topology": (
        grid_transform("targetWeightTransform", "targetWeightTransform", SPLIT_GRID_TOPOLOGY_TARGET,
                       shared_reward("rf", "topologyJump", define=False)),
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ),
    "grid_partial": grid_partial_transform("likelihood"),
    "grid_partial_kf": grid_partial_transform("kf"),
    # Pinned exponents, no tuner: optimise="false" makes startProposal()/optimize() no-ops. fixed2
    # is the reference point (see "pca_fixed2" above) - w^2 with zero offset, identical to the
    # no-transform path.
    **{f"fixed{k}": (
        '<targetWeightTransform id="targetWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="{b}" offset="0.0" optimise="false"/>',
        ' targetWeightTransform="@targetWeightTransform"',
        '<log idref="targetWeightTransform"/>',
    ) for k, b in FIXED_EXPONENTS.items()},
}

# Which transform variant each edge-weights setting is rendered with.
WEIGHT_TRANSFORM_FOR_SETTING = {
    "constant": "none",
    "parsimony": "none",
    "pca": "none",
    "pca_grid": "grid",
    "pca_grid_low": "grid_low",
    "pca_grid_mid": "grid_mid",
    **{f"pca_fixed{k}": f"fixed{k}" for k in FIXED_EXPONENTS if k in PCA_FIXED_EXPONENTS + ("2",)},
    "pca_nt": "none",
    "partial": "none",
    **{f"partial_fixed{k}": f"fixed{k}" for k in PARTIAL_FIXED_EXPONENTS},
    "pca_thr": "none",
    **{f"partial_thr_fixed{k}": f"fixed{k}" for k in PARTIAL_THREADED_EXPONENTS},
    **{f"partial_thr_src_{src}": "fixed0p5" for src in ("uniform", "pca")},
    **{f"pca_srclen{k}": "none" for k in LENGTH_EXPONENTS},
    **{f"partial_thr_srclen{k}": "fixed0p5" for k in LENGTH_EXPONENTS},
    "partial_thr_srclen_grid": "grid_partial",
    "partial_thr_srclen_grid_kf": "grid_partial_kf",
    **{k: "none" for k in ("pca_split_range12", "pca_split_slide", "pca_split_local", "pca_split_slide_tgt0p5", "pca_split_slide_tgt1", "pca_split_slide_tgt2")},
    "pca_split_grid": "grid_split_topology",
    "pca_split_grid_wide": "grid_split_topology_wide",
    **{f"pca_split_toff{k}": f"fixed3_off{k}" for k in TARGET_OFFSETS},
    **{f"pca_srcmix{k}": "none" for k in LENGTH_MIX_OFFSETS},
    **{f"partial_thr_srcmix{k}": "fixed0p5" for k in LENGTH_MIX_OFFSETS if k != "0"},
    "pca_grid_mid_rnni": "grid_mid_rnni",
    "pca_grid_mid_dual": "grid_mid_dual",
    "pca_grid_mid_ratio": "grid_mid_ratio",
    "pca_aos_mid": "none",
    "pca_l2": "none",
    **{f"pca_l2_fixed{k}": f"fixed{k}" for k in L2_EXPONENTS},
}

# edgeWeightTransform (source-edge weights, see targetedbeast.operators.TargetedWeightOperator) is
# a separate WeightTransform Input from targetWeightTransform on every operator that carries one -
# left unwired ("none") in every setting above, so source-edge selection has always used raw,
# untransformed weights. This "grid_mid" entry gives it its own PowerOffsetTransform/GridTuner,
# independent of (not shared with) targetWeightTransform's - same grid and reward as
# WEIGHT_TRANSFORM_XML's "grid_mid_dual" target transform, but its own id, so the two adapt
# separately and TargetedWeightOperator.optimize()/startTransformProposal() drive each once.
def length_exponent_edge_transform(exponent: str) -> tuple[str, str, str]:
    return (
        '<edgeWeightTransform id="edgeWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="{exponent}" offset="0.0" optimise="false"/>',
        ' edgeWeightTransform="@edgeWeightTransform"',
        '<log idref="edgeWeightTransform"/>',
    )


def length_grid_edge_transform(reward: str = "likelihood") -> tuple[str, str, str]:
    return (
        '<edgeWeightTransform id="edgeWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="1.0" offset="{LENGTH_GRID_OFFSET}" maxExponent="{max(GRID_EXPONENTS_LENGTH)}">\n'
        '            <tuner spec="targetedbeast.weighttransform.GridTuner"'
        f' exponents="{" ".join(str(c) for c in GRID_EXPONENTS_LENGTH)}"'
        f' explorationProb="0.1" burnin="{GRID_BURNIN}">\n'
        f'                {GRID_REWARDS[reward]}\n'
        '            </tuner>\n'
        '        </edgeWeightTransform>',
        ' edgeWeightTransform="@edgeWeightTransform"',
        '<log idref="edgeWeightTransform"/>',
    )


def length_mix_edge_transform(offset: str) -> tuple[str, str, str]:
    return (
        '<edgeWeightTransform id="edgeWeightTransform"'
        ' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
        f' exponent="1.0" offset="{offset}" offsetBase="branchLength" optimise="false"/>',
        ' edgeWeightTransform="@edgeWeightTransform"',
        '<log idref="edgeWeightTransform"/>',
    )


EDGE_WEIGHT_TRANSFORM_XML = {
    "none": ("", "", ""),
    **{f"lenmix{k}": length_mix_edge_transform(a) for k, a in LENGTH_MIX_OFFSETS.items()},
    **{f"lenexp{k}": length_exponent_edge_transform(c) for k, c in LENGTH_EXPONENTS.items()},
    "lengrid_split_topology_wide": (
        grid_transform("edgeWeightTransform", "edgeWeightTransform", SPLIT_GRID_WIDE_TOPOLOGY_SOURCE,
                       shared_reward("rf", "topologyJump", define=True), LENGTH_GRID_OFFSET),
        ' edgeWeightTransform="@edgeWeightTransform"',
        '<log idref="edgeWeightTransform"/>',
    ),
    "lengrid_split_topology": (
        grid_transform("edgeWeightTransform", "edgeWeightTransform", SPLIT_GRID_TOPOLOGY_SOURCE,
                       shared_reward("rf", "topologyJump", define=True), LENGTH_GRID_OFFSET),
        ' edgeWeightTransform="@edgeWeightTransform"',
        '<log idref="edgeWeightTransform"/>',
    ),
    "lengrid": length_grid_edge_transform(),
    "lengrid_kf": length_grid_edge_transform("kf"),
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
LIKELIHOOD_SPEC_FOR_SETTING = {
    "pca_nt": NON_THREADED_LIKELIHOOD_SPEC,
    "partial": NON_THREADED_LIKELIHOOD_SPEC,
    **{f"partial_fixed{k}": NON_THREADED_LIKELIHOOD_SPEC for k in PARTIAL_FIXED_EXPONENTS},
}

EDGE_WEIGHT_TRANSFORM_FOR_SETTING = {
    **{f"pca_srcmix{k}": f"lenmix{k}" for k in LENGTH_MIX_OFFSETS},
    **{f"pca_srclen{k}": f"lenexp{k}" for k in LENGTH_EXPONENTS},
    **{f"partial_thr_srclen{k}": f"lenexp{k}" for k in LENGTH_EXPONENTS},
    "partial_thr_srclen_grid": "lengrid",
    "partial_thr_srclen_grid_kf": "lengrid_kf",
    **{k: "lenexp0p5" for k in ("pca_split_range12", "pca_split_slide", "pca_split_local", "pca_split_slide_tgt0p5", "pca_split_slide_tgt1", "pca_split_slide_tgt2")},
    "pca_split_grid": "lengrid_split_topology",
    "pca_split_grid_wide": "lengrid_split_topology_wide",
    **{f"pca_split_toff{k}": "lenexp0p5" for k in TARGET_OFFSETS},
    **{f"partial_thr_srcmix{k}": f"lenmix{k}" for k in LENGTH_MIX_OFFSETS if k != "0"},
    "pca_grid_mid_dual": "grid_mid",
    "pca_grid_mid_ratio": "grid_mid_ratio",
    "pca_aos_mid": "none",
    "pca_l2": "none",
}


# Splitting height moves from topology moves (pca targets, length^0.5 sources for the topology
# operators). Overrides of the operator slots below, by setting:
#   pca_split_range12: plain Range at twice its weight;
#   pca_split_slide:   targetedRangeSlide's source exponent 1.5 (localEdgeWeightTransform's);
#   pca_split_local:   an added short weighted slide (localRangeSlide, size 0.5, source exponent 1.5).
def local_transform(role: str, exponent: str) -> str:
    """A fixed transform on localRangeSlide, by role ("edge" or "target")."""
    tag = f"{role}WeightTransform"
    return (f'<{tag} id="local{tag[0].upper()}{tag[1:]}" spec="targetedbeast.weighttransform.PowerOffsetTransform"'
            f' exponent="{exponent}" offset="0.0" optimise="false"/>')


SLIDE_TARGET_EXPONENTS = {"0p5": "0.5", "1": "1.0", "2": "2.0"}
SPLIT_SETTINGS = ("pca_split_range12", "pca_split_slide", "pca_split_local",
                  *(f"pca_split_slide_tgt{k}" for k in SLIDE_TARGET_EXPONENTS), "pca_split_grid",
                  "pca_split_grid_wide",
                  *TARGET_OFFSET_SETTINGS)
OPERATOR_SLOTS_FOR_SETTING = {
    "pca_split_range12": {"range_weight": "12.0"},
    "pca_split_slide": {"targeted_range_slide_edge_transform_ref": ' edgeWeightTransform="@localEdgeWeightTransform"'},
    "pca_split_local": {"local_range_slide_weight": "5.0"},
    # pca_split_slide with the slide's own target exponent b (1 is RangeSlide's untransformed default).
    **{f"pca_split_slide_tgt{k}": {
        "targeted_range_slide_edge_transform_ref": ' edgeWeightTransform="@localEdgeWeightTransform"',
        "targeted_range_slide_target_transform_ref": ' targetWeightTransform="@localTargetWeightTransform"',
        "local_target_weight_transform": local_transform("target", b),
    } for k, b in SLIDE_TARGET_EXPONENTS.items()},
    **{k: {
        "targeted_range_slide_edge_transform_ref": ' edgeWeightTransform="@localEdgeWeightTransform"',
        "targeted_range_slide_target_transform_ref": ' targetWeightTransform="@localTargetWeightTransform"',
    } for k in TARGET_OFFSET_SETTINGS},
    **{name: {
        "targeted_range_slide_edge_transform_ref": ' edgeWeightTransform="@localEdgeWeightTransform"',
        "targeted_range_slide_target_transform_ref": ' targetWeightTransform="@localTargetWeightTransform"',
        "local_edge_weight_transform": grid_transform("edgeWeightTransform", "localEdgeWeightTransform", source,
                                                      shared_reward("kf", "slideJump", define=True),
                                                      LENGTH_GRID_OFFSET),
        "local_target_weight_transform": grid_transform("targetWeightTransform", "localTargetWeightTransform",
                                                        SPLIT_GRID_SLIDE_TARGET,
                                                        shared_reward("kf", "slideJump", define=False)),
        "local_transform_log": '<log idref="localEdgeWeightTransform"/>\n            <log idref="localTargetWeightTransform"/>',
    } for name, source in (("pca_split_grid", SPLIT_GRID_SLIDE_SOURCE),
                           ("pca_split_grid_wide", SPLIT_GRID_WIDE_SLIDE_SOURCE))},
}


def operator_slots(setting: str, edge_transform_ref: str, target_transform_ref: str) -> dict[str, str]:
    slots = {
        "range_weight": "6.0",
        "targeted_range_slide_edge_transform_ref": edge_transform_ref,
        "targeted_range_slide_target_transform_ref": target_transform_ref,
        "local_range_slide_weight": "0.0",
        "local_transform_log": "",
        "local_edge_weight_transform": local_transform("edge", "1.5"),
        # RangeSlide uses target weights as they come, so exponent 1 is its untransformed default.
        "local_target_weight_transform": local_transform("target", "1.0"),
    }
    slots.update(OPERATOR_SLOTS_FOR_SETTING.get(setting, {}))
    return slots


def render_run_xml(template_text: str, setting: str, chain_length: int, out_dir: Path,
                    jump_interval: int, pre_burnin: int) -> str:
    transform_element, transform_ref, transform_log = \
        WEIGHT_TRANSFORM_XML[WEIGHT_TRANSFORM_FOR_SETTING[setting]]
    edge_transform_element, edge_transform_ref, edge_transform_log = \
        EDGE_WEIGHT_TRANSFORM_XML[EDGE_WEIGHT_TRANSFORM_FOR_SETTING.get(setting, "none")]
    xml = template_text.format(
        edge_weights=EDGE_WEIGHTS_XML[setting],
        chain_length=str(chain_length),
        target_weight_transform=transform_element,
        target_weight_transform_ref=transform_ref,
        target_weight_transform_log=transform_log,
        edge_weight_transform=edge_transform_element,
        edge_weight_transform_ref=edge_transform_ref,
        edge_weight_transform_log=edge_transform_log,
        likelihood_spec=LIKELIHOOD_SPEC_FOR_SETTING.get(setting, DEFAULT_LIKELIHOOD_SPEC),
        **operator_slots(setting, edge_transform_ref, transform_ref),
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
        # Per-operator split of what RNNI mixes together: root height, topology, branch lengths.
        '        <metric id="height" spec="targetedbeast.weighttransform.TreeHeightMetric"/>\n'
        '        <metric id="rf" spec="targetedbeast.weighttransform.RobinsonFouldsMetric"/>\n'
        '        <metric id="kf" spec="targetedbeast.weighttransform.KuhnerFelsensteinMetric"/>\n'
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

    for setting in args.settings:
        template_text = load_template(setting)
        for rep in range(args.reps):
            out_dir = args.out_root / setting / f"rep{rep}"
            out_dir.mkdir(parents=True, exist_ok=True)
            xml = render_run_xml(template_text, setting, args.chain_length, out_dir,
                                  args.jump_interval, args.pre_burnin)
            (out_dir / "run.xml").write_text(xml, encoding="utf-8")
            print(f"Wrote {out_dir / 'run.xml'}")


if __name__ == "__main__":
    main()
