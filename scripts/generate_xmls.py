from pathlib import Path
import argparse
import random


N_REPS = 4
CHAIN_LENGTH = 10_000_000
N_TAXA = None  # None = use the full alignment
SUBSAMPLE_SEED = 42
EXAMPLE_NAME = "lassaL_kli22"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_ROOT = PROJECT_ROOT / "examples" / EXAMPLE_NAME
TEMPLATE_PATH = EXAMPLE_ROOT / "template.xml"
ALIGNMENT_PATH = EXAMPLE_ROOT / "ali_normalized.fasta"


EDGE_WEIGHTS = {
	# 'untargeted': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
	# 'parsimony': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ParsimonyWeights2" tree="@tree" maxWeight="20" data="@alignment"/>',
	# 'partial': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PartialWeights" tree="@tree" data="@alignment" likelihood="@treeLikelihood"/>',
	# "learned": ' ',
# }
} | {
	f'pca_{d}': f'<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" tree="@tree" maxWeight="20" data="@alignment" dimension="{d}"/>'
    for d in [10]
# } | {
# 	f'brownian_{d}': f'<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeightsBrownian" tree="@tree" maxWeight="20" data="@alignment" dimension="{d}"/>'
#     for d in [15]
}

# The grid the GridTuner arms choose among. Geometric, because the exponent acts
# multiplicatively on the log weights. It spans the neutral setting (1, raw weights) and the
# historical default (2, squared) at the low end, and runs past WeightTransform's default ceiling
# of 20 at the high end, because a first sweep found the reward still climbing at 16. Points
# above the ceiling need maxExponent raised to match, or the tuner refuses to start.
GRID_EXPONENTS = [8, 12, 16, 24]

# Proposals discarded before any reward is accumulated. The class default of 1000 is far too
# small here: at 30k steps the chain is still descending thousands of log units per proposal, so
# every grid point measures the same transient and the curve is flat. targetedWilsonBalding sees
# roughly 1.3M of the 10M steps, so this is about 8%, in line with the ESS burn-in.
GRID_BURNIN = 10_000


def _grid_transform(exploration_prob: float) -> tuple[str, str, str]:
	"""A shared PowerOffsetTransform whose exponent is chosen from GRID_EXPONENTS.

	explorationProb 1.0 keeps the choice uniform forever, which measures the reward curve rather
	than optimizing it; below 1.0 it is the exploration floor. Running both checks whether the
	adaptive one lands where the sweep says it should - the falsifiable test SPSA failed.

	The reward is the squared change in likelihood per proposal, i.e. ESJD. Acceptance alone
	drove the exponent straight to the ceiling.
	"""
	return (
		'<targetWeightTransform id="targetWeightTransform"'
		' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
		f' exponent="2.0" offset="0.0" maxExponent="{max(GRID_EXPONENTS)}.0">\n'
		'                <tuner spec="targetedbeast.weighttransform.GridTuner"'
		f' exponents="{" ".join(str(b) for b in GRID_EXPONENTS)}"'
		f' explorationProb="{exploration_prob}" burnin="{GRID_BURNIN}">\n'
		'                    <reward idref="likelihood"/>\n'
		'                </tuner>\n'
		'            </targetWeightTransform>',
		' targetWeightTransform="@targetWeightTransform"',
		'<log idref="targetWeightTransform"/>',
	)


# Each entry is (transform element, reference attribute, tracelog XML). The element is declared
# once inside targetedWilsonBalding; the reference attribute is what the *other* targeted
# operators use to point at that same object, so a single set of coefficients is shared and all
# their observations feed one adaptation. The 'none' entry leaves every operator on its
# historical default, which is the baseline every comparison is against.
WEIGHT_TRANSFORMS = {
	'none': ('', '', ''),
	# Exponent tuned by SPSA, offset pinned at zero: the exponent dominates the proposal shape,
	# and tuning one coefficient rather than two roughly halves the gradient noise.
	# 'power': (
	# 	'<targetWeightTransform id="targetWeightTransform"'
	# 	' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
	# 	' exponent="2.0" offset="0.0"/>',
	# 	' targetWeightTransform="@targetWeightTransform"',
	# 	'<log idref="targetWeightTransform"/>',
	# ),
	# The same coefficient, searched over a fixed grid instead of by local perturbation. The
	# sweep and the adaptive variant differ only in explorationProb, so they share a builder.
	# 'grid_sweep': _grid_transform(exploration_prob=1.0),
	'grid': _grid_transform(exploration_prob=0.1),
}

# Edge weights that are uniform over candidates, so a transform cannot change the proposal:
# any transform of a constant vector is still constant once normalized. Running these against
# every transform would just burn CPU on identical baselines.
UNIFORM_EDGE_WEIGHTS = {'untargeted'}


def transforms_for(weights_key: str) -> list[str]:
	"""Transform variants worth running for one edge-weights entry."""
	if weights_key in UNIFORM_EDGE_WEIGHTS:
		return ['none']
	return list(WEIGHT_TRANSFORMS)


# One subdirectory per (edge weights, transform) combination actually worth running.
SUBDIRS = [f"{w}__{t}" for w in EDGE_WEIGHTS for t in transforms_for(w)]


def read_fasta(path: Path) -> list[tuple[str, str]]:
	"""Parse a FASTA file into (header, sequence) pairs, joining wrapped sequence lines."""
	records: list[tuple[str, str]] = []
	header: str | None = None
	chunks: list[str] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		if line.startswith(">"):
			if header is not None:
				records.append((header, "".join(chunks)))
			header = line[1:].strip()
			chunks = []
		elif line.strip():
			chunks.append(line.strip())
	if header is not None:
		records.append((header, "".join(chunks)))
	return records


def subsampled_alignment(n_taxa: int | None) -> Path:
	"""
	Return the alignment to point the XML at, writing a subsampled copy if asked.

	Tip dates are parsed out of the taxon names by the template (delimiter="|"), so a
	subsample needs nothing but a subset of the records - there is no separate date file
	to keep in sync. The draw is seeded, so a given n_taxa always yields the same taxa and
	runs stay comparable across invocations.
	"""
	if n_taxa is None:
		return ALIGNMENT_PATH

	records = read_fasta(ALIGNMENT_PATH)
	if n_taxa > len(records):
		raise SystemExit(f"--n-taxa {n_taxa} exceeds the {len(records)} taxa in {ALIGNMENT_PATH.name}")
	if n_taxa < 4:
		raise SystemExit("--n-taxa must be at least 4 for a tree operator to have anything to do")

	chosen = random.Random(SUBSAMPLE_SEED).sample(records, n_taxa)
	# Keep the original file order, so the subsample differs from the full set only by
	# which taxa are present.
	order = {header: i for i, (header, _) in enumerate(records)}
	chosen.sort(key=lambda r: order[r[0]])

	out_path = EXAMPLE_ROOT / f"ali_normalized.n{n_taxa}.fasta"
	out_path.write_text(
		"".join(f">{header}\n{seq}\n" for header, seq in chosen),
		encoding="utf-8",
	)
	print(f"Wrote {out_path} ({n_taxa} of {len(records)} taxa)")
	return out_path


def render_xml(
	template_text: str,
	weights_string: str,
	transform: tuple[str, str, str],
	chain_length: int,
	alignment_file: Path,
) -> str:
	transform_element, transform_ref, transform_log = transform
	return template_text.format(
		edge_weights=weights_string,
		target_weight_transform=transform_element,
		target_weight_transform_ref=transform_ref,
		target_weight_transform_log=transform_log,
		chain_length=str(chain_length),
		alignment_file=str(alignment_file),
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Render the BEAST XML files from the template.")
	parser.add_argument(
		"--n-reps",
		type=int,
		default=N_REPS,
		help="number of replicate run?.xml files to generate per subdirectory",
	)
	parser.add_argument(
		"--chain-length",
		type=int,
		default=CHAIN_LENGTH,
		help="MCMC chain length; lower this for a quick development comparison",
	)
	parser.add_argument(
		"--n-taxa",
		type=int,
		default=N_TAXA,
		help="subsample the alignment to this many taxa; omit to use all of them. "
			"Per-iteration cost grows with the tree size, so this is the main knob for "
			"trading accuracy against runtime",
	)
	args = parser.parse_args()

	template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
	alignment_file = subsampled_alignment(args.n_taxa)

	for w, weights_string in EDGE_WEIGHTS.items():
		for t in transforms_for(w):
			transform = WEIGHT_TRANSFORMS[t]
			rendered_xml = render_xml(
				template_text, weights_string, transform, args.chain_length, alignment_file)

			output_dir = EXAMPLE_ROOT / f"{w}__{t}"
			output_dir.mkdir(parents=True, exist_ok=True)

			for i in range(args.n_reps):
				output_path = output_dir / f"run{i}.xml"
				output_path.write_text(rendered_xml, encoding="utf-8")
				print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()

