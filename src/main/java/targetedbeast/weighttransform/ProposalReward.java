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

	/** Starts a proposal on any reward Function; returns the "before" value of a plain Function. */
	static double begin(Function reward) {
		if (reward instanceof ProposalReward r) {
			r.beginProposal();
			return 0.0;
		}
		return reward.getArrayValue();
	}

	/** The jump across the proposal begun with {@link #begin}, given the value that returned. */
	static double jump(Function reward, double before) {
		return reward instanceof ProposalReward r ? r.jump() : reward.getArrayValue() - before;
	}
}
