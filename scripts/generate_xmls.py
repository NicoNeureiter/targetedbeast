from pathlib import Path
import argparse


N_REPS = 4
CHAIN_LENGTH = 10_000_000
EXAMPLE_NAME = "lassaL_kli22"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_ROOT = PROJECT_ROOT / "examples" / EXAMPLE_NAME
TEMPLATE_PATH = EXAMPLE_ROOT / "template.xml"


EDGE_WEIGHTS = {
	'untargeted': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ConstantWeights" tree="@tree"/>',
	'parsimony': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.ParsimonyWeights2" tree="@tree" maxWeight="20" data="@alignment"/>',
	# 'partial': '<distribution id="edgeweights" spec="targetedbeast.edgeweights.PartialWeights" tree="@tree" data="@alignment" likelihood="@treeLikelihood"/>',
	# "learned": ' ',
# }
# } | {
# 	f'pca_{d}': f'<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeights2" tree="@tree" maxWeight="20" data="@alignment" dimension="{d}"/>'
#     for d in [3, 6, 9]
} | {
	f'brownian_{d}': f'<distribution id="edgeweights" spec="targetedbeast.edgeweights.PCAWeightsBrownian" tree="@tree" maxWeight="20" data="@alignment" dimension="{d}"/>'
    for d in [3, 6]
}

# Each entry is (operator-nested XML, tracelog XML). The 'none' entry keeps the operator
# on its historical hardcoded squaring, which is the baseline every comparison is against.
WEIGHT_TRANSFORMS = {
	'none': ('', ''),
	'power': (
		'<targetWeightTransform id="targetWeightTransform"'
		' spec="targetedbeast.weighttransform.PowerOffsetTransform"'
		' exponent="2.0" offset="0.01"/>',
		'<log idref="targetWeightTransform"/>',
	),
}

# One subdirectory per (edge weights, transform) combination.
SUBDIRS = [f"{w}__{t}" for w in EDGE_WEIGHTS for t in WEIGHT_TRANSFORMS]


def render_xml(template_text: str, weights_string: str, transform: tuple[str, str], chain_length: int) -> str:
	transform_xml, transform_log = transform
	return template_text.format(
		edge_weights=weights_string,
		target_weight_transform=transform_xml,
		target_weight_transform_log=transform_log,
		chain_length=str(chain_length),
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
	args = parser.parse_args()

	template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

	for w, weights_string in EDGE_WEIGHTS.items():
		for t, transform in WEIGHT_TRANSFORMS.items():
			rendered_xml = render_xml(template_text, weights_string, transform, args.chain_length)

			output_dir = EXAMPLE_ROOT / f"{w}__{t}"
			output_dir.mkdir(parents=True, exist_ok=True)

			for i in range(args.n_reps):
				output_path = output_dir / f"run{i}.xml"
				output_path.write_text(rendered_xml, encoding="utf-8")
				print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()

