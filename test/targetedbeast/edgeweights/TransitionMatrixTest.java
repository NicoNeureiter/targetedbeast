package targetedbeast.edgeweights;

import static org.junit.Assert.*;

import java.util.Arrays;
import org.junit.Before;
import org.junit.Test;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.parameter.RealParameter;
import beast.base.evolution.likelihood.TreeLikelihood;

/**
 * Fine-grained unit tests for applyTransitionMatrix and applyTransposeTransitionMatrix
 * methods in FelsensteinWeights.
 * 
 * These tests focus on:
 * 1. Correct matrix-vector multiplication logic
 * 2. Proper indexing with multiple patterns and rate categories
 * 3. Inverse relationship between forward and transpose operations
 * 4. Numerical accuracy
 */
public class TransitionMatrixTest {

	private Alignment alignment;
	private Tree tree;
	private TreeLikelihood likelihood;
	private SiteModel siteModel;
	private FelsensteinWeights weights;

	/**
	 * Set up a simple 4-taxon tree with JC69 model and 4 gamma categories.
	 */
	@Before
	public void setUp() {
		// Create alignment with 4 taxa
		Sequence seqA = new Sequence("A", "ACGTACGTAA");
		Sequence seqB = new Sequence("B", "ACGTACGTAG");
		Sequence seqC = new Sequence("C", "ACGTAAGTAC");
		Sequence seqD = new Sequence("D", "ACGTAAGTAC");

		alignment = new Alignment();
		alignment.initByName("sequence", seqA, "sequence", seqB, 
							 "sequence", seqC, "sequence", seqD);

		// Create tree
		tree = new TreeParser();
		((TreeParser) tree).initByName(
			"newick", "((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.15);",
			"IsLabelledNewick", true,
			"taxa", alignment
		);

		// Create substitution model
		JukesCantor jc = new JukesCantor();
		jc.initAndValidate();
        
        RealParameter shapeParameter = new RealParameter("1.0");

		// Create site model with 4 gamma categories
		siteModel = new SiteModel();
		siteModel.initByName("substModel", jc, "gammaCategoryCount", 4, "shape", shapeParameter);

		// Create likelihood
		System.setProperty("java.only", "true");
		likelihood = new TreeLikelihood();
		likelihood.initByName("data", alignment, "tree", tree, "siteModel", siteModel);

		// Create FelsensteinWeights
		weights = new FelsensteinWeights();
		weights.initByName(
			"data", alignment,
			"tree", tree,
			"likelihood", likelihood,
			"minWeight", 0.01,
			"maxWeight", 10.0
		);
        weights.ensureInitialized();
	}

	// ========== Forward Apply Tests ==========

	/**
	 * Test applyTransitionMatrix with identity matrix and uniform partials.
	 * Expected: output should equal input (identity operation).
	 */
	@Test
	public void testApplyTransitionMatrixIdentity() {
		likelihood.calculateLogP();

		// Create identity matrix for a single category
		// For JC69 with 4 states: I = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
		double[] identityMatrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);

		// Create simple test partials (uniform)
		double[] testPartials = createUniformPartials(0.25);

		// Apply identity matrix
		double[] result = weights.applyTransitionMatrix(identityMatrix, testPartials);

		// Result should equal input
		for (int i = 0; i < testPartials.length; i++) {
			assertEquals("Identity matrix should preserve partials at index " + i,
					testPartials[i], result[i], 1e-10);
		}
	}

	/**
	 * Test applyTransitionMatrix with zero matrix.
	 * Expected: all outputs should be zero (or very close due to numerical errors).
	 */
	@Test
	public void testApplyTransitionMatrixZero() {
		likelihood.calculateLogP();

		// Create zero matrix
		double[] zeroMatrix = new double[weights.matrixSize * weights.matrixCount];
		// All zeros by default

		// Create test partials
		double[] testPartials = createUniformPartials(1.0);

		// Apply zero matrix
		double[] result = weights.applyTransitionMatrix(zeroMatrix, testPartials);

		// All results should be zero
		for (int i = 0; i < result.length; i++) {
			assertEquals("Zero matrix should produce zero output at index " + i,
					0.0, result[i], 1e-10);
		}
	}

	/**
	 * Test applyTransitionMatrix with diagonal matrix (scaling operation).
	 * For a diagonal matrix D = [[a,0,0,0], [0,b,0,0], [0,0,c,0], [0,0,0,d]],
	 * applying to [p1, p2, p3, p4] should yield [a*p1, b*p2, c*p3, d*p4].
	 */
	@Test
	public void testApplyTransitionMatrixDiagonal() {
		likelihood.calculateLogP();

		// Create diagonal matrix with specific scaling factors
		double[] diagonalMatrix = createDiagonalMatrix(
			new double[]{2.0, 3.0, 4.0, 5.0}, 
			weights.stateCount, 
			weights.matrixCount
		);

		// Create test partials with known values
		double[] testPartials = new double[weights.partialsSize];
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[i] = (double)(i + 1); // [1, 2, 3, 4]
		}

		double[] result = weights.applyTransitionMatrix(diagonalMatrix, testPartials);

		// Check first pattern, first category
		assertEquals("Diagonal scaling for state 0", 2.0 * 1.0, result[0], 1e-10);
		assertEquals("Diagonal scaling for state 1", 3.0 * 2.0, result[1], 1e-10);
		assertEquals("Diagonal scaling for state 2", 4.0 * 3.0, result[2], 1e-10);
		assertEquals("Diagonal scaling for state 3", 5.0 * 4.0, result[3], 1e-10);
	}

	/**
	 * Test applyTransitionMatrix with two patterns.
	 * Verify that patterns are processed independently.
	 */
	@Test
	public void testApplyTransitionMatrixMultiplePatterns() {
		likelihood.calculateLogP();

		double[] identityMatrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);

		// Create partials with different values for two patterns
		double[] testPartials = new double[weights.partialsSize];
		// Pattern 0: [1.0, 2.0, 3.0, 4.0]
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[i] = i + 1.0;
		}
		// Pattern 1: [5.0, 6.0, 7.0, 8.0]
		int offset1 = weights.getPartialOffset(1, 0);
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[offset1 + i] = i + 5.0;
		}

		double[] result = weights.applyTransitionMatrix(identityMatrix, testPartials);

		// Verify pattern 0
		for (int i = 0; i < weights.stateCount; i++) {
			assertEquals("Pattern 0, state " + i,
					testPartials[i], result[i], 1e-10);
		}

		// Verify pattern 1
		for (int i = 0; i < weights.stateCount; i++) {
			assertEquals("Pattern 1, state " + i,
					testPartials[offset1 + i], result[offset1 + i], 1e-10);
		}
	}

	/**
	 * Test applyTransitionMatrix with multiple rate categories.
	 * Verify that categories are processed independently.
	 */
	@Test
	public void testApplyTransitionMatrixMultipleCategories() {
		likelihood.calculateLogP();

		// Create identity matrices for 4 categories
		double[] identityMatrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);

		// Create partials where each category has different values
		double[] testPartials = new double[weights.partialsSize];
		for (int category = 0; category < weights.matrixCount; category++) {
			int offset = category * weights.patternCount * weights.stateCount;
			for (int i = 0; i < weights.stateCount; i++) {
				testPartials[offset + i] = category * 10.0 + i;
			}
		}

		double[] result = weights.applyTransitionMatrix(identityMatrix, testPartials);

		// Verify each category
		for (int category = 0; category < weights.matrixCount; category++) {
			int offset = category * weights.patternCount * weights.stateCount;
			for (int i = 0; i < weights.stateCount; i++) {
				assertEquals("Category " + category + ", state " + i,
						testPartials[offset + i], result[offset + i], 1e-10);
			}
		}
	}

	/**
	 * Test applyTransitionMatrix(int childNodeNr, double[], double[]) variant.
	 * Should use node matrices from likelihood core.
	 */
	@Test
	public void testApplyTransitionMatrixWithNodeNr() {
		likelihood.calculateLogP();

		// Get a non-root node
		int childNr = 0; // First node (likely a leaf)

		// Create test partials
		double[] testPartials = createUniformPartials(1.0);
		double[] result = new double[testPartials.length];

		// Should not throw
		weights.applyTransitionMatrix(childNr, testPartials, result);

		// Result should have valid values (not NaN or infinite)
		for (int i = 0; i < result.length; i++) {
			assertTrue("Result should be finite at index " + i, Double.isFinite(result[i]));
		}
	}

	/**
	 * Test applyTransitionMatrix with very small input values.
	 * Verify numerical stability.
	 */
	@Test
	public void testApplyTransitionMatrixSmallValues() {
		likelihood.calculateLogP();

		double[] identityMatrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);

		// Create partials with very small values
		double[] testPartials = createUniformPartials(1e-100);

		double[] result = weights.applyTransitionMatrix(identityMatrix, testPartials);

		// Small values should be preserved
		for (int i = 0; i < testPartials.length; i++) {
			assertEquals("Small values should be preserved",
					testPartials[i], result[i], 1e-110);
		}
	}

	// ========== Transpose Apply Tests ==========

	/**
	 * Test applyTransposeTransitionMatrix with identity matrix.
	 * Expected: output should equal input (identity is symmetric).
	 */
	@Test
	public void testApplyTransposeIdentity() {
		likelihood.calculateLogP();

		double[] identityMatrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);
		double[] testPartials = createUniformPartials(0.25);

		double[] result = weights.applyTransposeTransitionMatrix(identityMatrix, testPartials);

		for (int i = 0; i < testPartials.length; i++) {
			assertEquals("Transpose of identity should preserve partials at " + i,
					testPartials[i], result[i], 1e-10);
		}
	}

	/**
	 * Test applyTransposeTransitionMatrix with diagonal matrix.
	 * Diagonal matrices are symmetric, so transpose should give same result as forward.
	 */
	@Test
	public void testApplyTransposeDiagonal() {
		likelihood.calculateLogP();

		double[] diagonalMatrix = createDiagonalMatrix(
			new double[]{2.0, 3.0, 4.0, 5.0},
			weights.stateCount,
			weights.matrixCount
		);

		double[] testPartials = new double[weights.partialsSize];
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[i] = (double)(i + 1);
		}

		// Forward and transpose should give same result for diagonal
		double[] forward = weights.applyTransitionMatrix(diagonalMatrix, testPartials);
		double[] transpose = weights.applyTransposeTransitionMatrix(diagonalMatrix, testPartials);

		for (int i = 0; i < forward.length; i++) {
			assertEquals("Diagonal matrix is symmetric, forward and transpose should match at " + i,
					forward[i], transpose[i], 1e-10);
		}
	}

	/**
	 * Test applyTransposeTransitionMatrix with zero matrix.
	 */
	@Test
	public void testApplyTransposeZero() {
		likelihood.calculateLogP();

		double[] zeroMatrix = new double[weights.matrixSize * weights.matrixCount];
		double[] testPartials = createUniformPartials(1.0);

		double[] result = weights.applyTransposeTransitionMatrix(zeroMatrix, testPartials);

		for (int i = 0; i < result.length; i++) {
			assertEquals("Transpose of zero matrix should produce zero", 0.0, result[i], 1e-10);
		}
	}

	/**
	 * Test applyTransposeTransitionMatrix with permutation matrix.
	 * A permutation matrix P has exactly one 1 in each row and column.
	 * P^T swaps rows and columns.
	 */
	@Test
	public void testApplyTransposePermutation() {
		likelihood.calculateLogP();

		// Create a permutation matrix that cycles: 0->1->2->3->0
		double[] permMatrix = createPermutationMatrix(
			new int[]{1, 2, 3, 0},  // permutation: 0->1, 1->2, 2->3, 3->0
			weights.stateCount,
			weights.matrixCount
		);

		double[] testPartials = new double[weights.partialsSize];
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[i] = (double)(i + 1); // [1, 2, 3, 4]
		}

		// Apply transpose
		double[] result = weights.applyTransposeTransitionMatrix(permMatrix, testPartials);

		// P^T cycles in reverse: 0<-1<-2<-3<-0, so:
		// result[0] = input[3] = 4
		// result[1] = input[0] = 1
		// result[2] = input[1] = 2
		// result[3] = input[2] = 3
		assertEquals("Transpose permutation state 0", 4.0, result[0], 1e-10);
		assertEquals("Transpose permutation state 1", 1.0, result[1], 1e-10);
		assertEquals("Transpose permutation state 2", 2.0, result[2], 1e-10);
		assertEquals("Transpose permutation state 3", 3.0, result[3], 1e-10);
	}

	/**
	 * Test applyTransposeTransitionMatrix with multiple patterns.
	 */
	@Test
	public void testApplyTransposeMultiplePatterns() {
		likelihood.calculateLogP();

		double[] identityMatrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);

		double[] testPartials = new double[weights.partialsSize];
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[i] = i + 1.0;
		}
		int offset1 = weights.getPartialOffset(1, 0);
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[offset1 + i] = i + 5.0;
		}

		double[] result = weights.applyTransposeTransitionMatrix(identityMatrix, testPartials);

		for (int i = 0; i < weights.stateCount; i++) {
			assertEquals("Pattern 0, state " + i,
					testPartials[i], result[i], 1e-10);
		}
		for (int i = 0; i < weights.stateCount; i++) {
			assertEquals("Pattern 1, state " + i,
					testPartials[offset1 + i], result[offset1 + i], 1e-10);
		}
	}

	/**
	 * Test applyTransposeTransitionMatrix with multiple rate categories.
	 */
	@Test
	public void testApplyTransposeMultipleCategories() {
		likelihood.calculateLogP();

		double[] identityMatrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);

		double[] testPartials = new double[weights.partialsSize];
		for (int category = 0; category < weights.matrixCount; category++) {
			int offset = category * weights.patternCount * weights.stateCount;
			for (int i = 0; i < weights.stateCount; i++) {
				testPartials[offset + i] = category * 10.0 + i;
			}
		}

		double[] result = weights.applyTransposeTransitionMatrix(identityMatrix, testPartials);

		for (int category = 0; category < weights.matrixCount; category++) {
			int offset = category * weights.patternCount * weights.stateCount;
			for (int i = 0; i < weights.stateCount; i++) {
				assertEquals("Category " + category + ", state " + i,
						testPartials[offset + i], result[offset + i], 1e-10);
			}
		}
	}

	/**
	 * Test applyTransposeTransitionMatrix(int childNodeNr, ...) variant.
	 */
	@Test
	public void testApplyTransposeWithNodeNr() {
		likelihood.calculateLogP();

		int childNr = 0;
		double[] testPartials = createUniformPartials(1.0);
		double[] result = new double[testPartials.length];

		// Should not throw
		weights.applyTransposeTransitionMatrix(childNr, testPartials, result);

		for (int i = 0; i < result.length; i++) {
			assertTrue("Result should be finite at index " + i, Double.isFinite(result[i]));
		}
	}

	// ========== Relationship Tests ==========

	/**
	 * Test that applyTransitionMatrix and applyTransposeTransitionMatrix are inverses
	 * for orthogonal (symmetric positive definite) matrices.
	 * For an orthogonal matrix M: M * M^T = I
	 */
	@Test
	public void testForwardTransposeCompositionIdentity() {
		likelihood.calculateLogP();

		// Use identity matrix (which is orthogonal)
		double[] matrix = createIdentityMatrix(weights.stateCount, weights.matrixCount);
		double[] testPartials = createUniformPartials(0.5);

		// Apply M then M^T
		double[] afterForward = weights.applyTransitionMatrix(matrix, testPartials);
		double[] afterBoth = weights.applyTransposeTransitionMatrix(matrix, afterForward);

		// Should recover original (approximately)
		for (int i = 0; i < testPartials.length; i++) {
			assertEquals("M * M^T should be identity at " + i,
					testPartials[i], afterBoth[i], 1e-10);
		}
	}

	/**
	 * Test that forward and transpose output different results for non-symmetric matrices.
	 * This verifies they actually compute different things.
	 */
	@Test
	public void testForwardTransposeDifferent() {
		likelihood.calculateLogP();

		// Create an asymmetric matrix
		double[] asymmetricMatrix = new double[weights.matrixSize * weights.matrixCount];
		// Fill first category with an asymmetric matrix
		// [[1, 2, 0, 0],
		//  [0, 1, 3, 0],
		//  [0, 0, 1, 4],
		//  [0, 0, 0, 1]]
		int i = 0;
		asymmetricMatrix[i++] = 1; asymmetricMatrix[i++] = 2; asymmetricMatrix[i++] = 0; asymmetricMatrix[i++] = 0;
		asymmetricMatrix[i++] = 0; asymmetricMatrix[i++] = 1; asymmetricMatrix[i++] = 3; asymmetricMatrix[i++] = 0;
		asymmetricMatrix[i++] = 0; asymmetricMatrix[i++] = 0; asymmetricMatrix[i++] = 1; asymmetricMatrix[i++] = 4;
		asymmetricMatrix[i++] = 0; asymmetricMatrix[i++] = 0; asymmetricMatrix[i++] = 0; asymmetricMatrix[i++] = 1;

		double[] testPartials = createUniformPartials(1.0);

		double[] forward = weights.applyTransitionMatrix(asymmetricMatrix, testPartials);
		double[] transpose = weights.applyTransposeTransitionMatrix(asymmetricMatrix, testPartials);

		// They should be different (for non-symmetric matrix)
		double maxDiff = 0.0;
		for (int j = 0; j < forward.length; j++) {
			maxDiff = Math.max(maxDiff, Math.abs(forward[j] - transpose[j]));
		}
		assertTrue("Forward and transpose should differ for non-symmetric matrix",
				maxDiff > 1e-10);
	}

	// ========== Numerical Accuracy Tests ==========

	/**
	 * Test applyTransitionMatrix preserves row stochasticity property.
	 * For a stochastic matrix, row sums should be 1.0.
	 * When applied to uniform partials, output should also be uniform.
	 */
	@Test
	public void testApplyTransitionMatrixRowStochasticity() {
		likelihood.calculateLogP();

		// Create a row-stochastic matrix
		double[] stochasticMatrix = createRowStochasticMatrix(weights.stateCount, weights.matrixCount);

		double[] uniformPartials = createUniformPartials(0.25);
		double[] result = weights.applyTransitionMatrix(stochasticMatrix, uniformPartials);

		// Output should be approximately uniform
		for (int i = 0; i < result.length; i++) {
			assertEquals("Row-stochastic matrix applied to uniform should give uniform at " + i,
					0.25, result[i], 1e-10);
		}
	}

	/**
	 * Test applyTransitionMatrix linearity: M * (a*v) = a * (M*v)
	 */
	@Test
	public void testApplyTransitionMatrixLinearity() {
		likelihood.calculateLogP();

		double[] matrix = createDiagonalMatrix(
			new double[]{2.0, 3.0, 4.0, 5.0},
			weights.stateCount,
			weights.matrixCount
		);

		double[] testPartials = new double[weights.partialsSize];
		for (int i = 0; i < weights.stateCount; i++) {
			testPartials[i] = i + 1.0;
		}

		// Compute M * v
		double[] mv = weights.applyTransitionMatrix(matrix, testPartials);

		// Compute M * (2*v)
		double[] scaled = new double[testPartials.length];
		for (int i = 0; i < testPartials.length; i++) {
			scaled[i] = 2.0 * testPartials[i];
		}
		double[] m2v = weights.applyTransitionMatrix(matrix, scaled);

		// m2v should equal 2 * mv
		for (int i = 0; i < mv.length; i++) {
			assertEquals("Linearity test at " + i,
					2.0 * mv[i], m2v[i], 1e-10);
		}
	}

	// ========== Helper Methods ==========

	private double[] createIdentityMatrix(int stateCount, int categoryCount) {
		double[] matrix = new double[stateCount * stateCount * categoryCount];
		for (int cat = 0; cat < categoryCount; cat++) {
			int catOffset = cat * stateCount * stateCount;
			for (int i = 0; i < stateCount; i++) {
				matrix[catOffset + i * stateCount + i] = 1.0;
			}
		}
		return matrix;
	}

	private double[] createDiagonalMatrix(double[] diagonalValues, int stateCount, int categoryCount) {
		double[] matrix = new double[stateCount * stateCount * categoryCount];
		for (int cat = 0; cat < categoryCount; cat++) {
			int catOffset = cat * stateCount * stateCount;
			for (int i = 0; i < stateCount; i++) {
				matrix[catOffset + i * stateCount + i] = diagonalValues[i];
			}
		}
		return matrix;
	}

	private double[] createPermutationMatrix(int[] permutation, int stateCount, int categoryCount) {
		double[] matrix = new double[stateCount * stateCount * categoryCount];
		for (int cat = 0; cat < categoryCount; cat++) {
			int catOffset = cat * stateCount * stateCount;
			for (int i = 0; i < stateCount; i++) {
				int j = permutation[i];
				matrix[catOffset + i * stateCount + j] = 1.0;
			}
		}
		return matrix;
	}

	private double[] createRowStochasticMatrix(int stateCount, int categoryCount) {
		double[] matrix = new double[stateCount * stateCount * categoryCount];
		for (int cat = 0; cat < categoryCount; cat++) {
			int catOffset = cat * stateCount * stateCount;
			for (int i = 0; i < stateCount; i++) {
				for (int j = 0; j < stateCount; j++) {
					matrix[catOffset + i * stateCount + j] = 1.0 / stateCount;
				}
			}
		}
		return matrix;
	}

	private double[] createUniformPartials(double value) {
		double[] partials = new double[weights.partialsSize];
		Arrays.fill(partials, value);
		return partials;
	}
}
