package targetedbeast.weighttransform;

import beast.base.core.Function;

/**
 * A tuner reward measured across one proposal, for quantities that are not a value of the state,
 * such as a tree-distance jump. The tuner calls {@link #beginProposal()} when a proposal starts and
 * {@link #jump()} when MCMC reports its outcome. MCMC does not report every proposal (it skips
 * optimize() during pre-burnin and on logging steps), so an implementation must not assume the two
 * calls alternate: a new beginProposal() simply starts over.
 */
public interface ProposalReward {

	void beginProposal();

	/** Size of the change since the last {@link #beginProposal()}. */
	double jump();

	/** The tuner's reward for the proposal: by default the squared jump; may be signed. */
	default double reward() {
		double jump = jump();
		return jump * jump;
	}

	/** Starts a proposal on any reward Function; returns the "before" value of a plain Function. */
	static double begin(Function reward) {
		if (reward instanceof ProposalReward r) {
			r.beginProposal();
			return 0.0;
		}
		return reward.getArrayValue();
	}

	/** The reward for the proposal begun with {@link #begin}, given the value that returned. */
	static double reward(Function reward, double before) {
		if (reward instanceof ProposalReward r) {
			return r.reward();
		}
		double jump = reward.getArrayValue() - before;
		return jump * jump;
	}
}
