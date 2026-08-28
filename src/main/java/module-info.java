open module targetedbeast {
    requires beast.pkgmgmt;
    requires beast.base;
    // targetedbeast.annotator.CladeAnnotator, targetedbeast.alignment.ConsensusAlignment
    requires beast.fx;

    requires java.desktop;               // JOptionPane in targetedbeast.annotator
    requires colt;                       // SVD in targetedbeast.util.Alignment2PCA
    requires commons.math3;              // linear algebra in targetedbeast.edgeweights.PCAWeights

    exports targetedbeast.alignment;
    exports targetedbeast.altoperators;
    exports targetedbeast.annotator;
    exports targetedbeast.coalescent;
    exports targetedbeast.edgeweights;
    exports targetedbeast.init;
    exports targetedbeast.likelihood;
    exports targetedbeast.logger;
    exports targetedbeast.operators;
    exports targetedbeast.operatorschedule;
    exports targetedbeast.util;

    provides beast.base.core.BEASTInterface with
        targetedbeast.alignment.ConsensusAlignment,
        targetedbeast.alignment.RapidAlignment,
        targetedbeast.altoperators.LengthConsistentUpDown,
        targetedbeast.altoperators.SubtreeSlide2,
        targetedbeast.altoperators.SubtreeSlide5,
        targetedbeast.altoperators.Uni,
        targetedbeast.altoperators.Uniform,
        targetedbeast.altoperators.Uniform2,
        targetedbeast.altoperators.WilsonBalding2,
        targetedbeast.coalescent.RapidCoalescent,
        targetedbeast.coalescent.RapidTreeIntervals,
        targetedbeast.edgeweights.CombinedEdgeWeights,
        targetedbeast.edgeweights.ConsensusWeights,
        targetedbeast.edgeweights.ConstantWeights,
        targetedbeast.edgeweights.FelsensteinWeights,
        targetedbeast.edgeweights.PCAWeights,
        targetedbeast.edgeweights.ParsimonyWeights,
        targetedbeast.edgeweights.ParsimonyWeights2,
        targetedbeast.init.ensureNoZeroEdgeLength,
        targetedbeast.likelihood.RapidGenericTreeLikelihood,
        targetedbeast.likelihood.RapidTreeLikelihood,
        targetedbeast.likelihood.SlowTreeLikelihood,
        targetedbeast.logger.CladeLogger,
        targetedbeast.operators.AdaptableOperatorSampler,
        targetedbeast.operators.BactrianIntervalScaleOperator,
        targetedbeast.operators.HeightBasedNodeRandomizer,
        targetedbeast.operators.IntervalScaleOperator,
        targetedbeast.operators.NodeRandomizer2,
        targetedbeast.operators.NodeRandomizer22,
        targetedbeast.operators.RangeSlide,
        targetedbeast.operators.TargetedWilsonBalding,
        targetedbeast.operators.WeightBasedNodeRandomizer,
        targetedbeast.operators.WeightedWideOperator,
        targetedbeast.operatorschedule.TargetedOperatorSchedule,
        targetedbeast.util.Alignment2PCA;
}
