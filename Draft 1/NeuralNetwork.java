import java.util.HashMap;
import java.util.ArrayList;
import java.lang.Math;

// LOOK INTO STOCHASTIC GRADIENT DESCENT.
/**
 * Basic implementation of a feed-forward neural network that uses the back-propagation algorithm with supervised learning. The implementation is generalized and 
 * handles arbitrary numbers of neurons, layers, training sets, etc.
 */
public class NeuralNetwork {
	/**
	 * Class that represents a single neuron, ie a basic processing unit, in this network.
	 */
	public static class Neuron {
		HashMap<Neuron, Double> synapses; // List of all the neurons to which this neuron is connected, along with the weight associated with that connection.
		ArrayList<Input> inputs_weights; // All the inputs received from the neurons in the previous layer, as well as the corresponding weights.
		boolean thresholdNeuron = false; // Whether or not this neuron is a filler neuron used for threshold purposes.
		boolean inputNeuron = false; // Whether or not this neuron is an input neuron.

		/**
		 * The activation function for this neuron, which is simply the dot product of the input vector (list of all inputs into this neuron) and the weight
		 * vector (corresponding list of all weights from the neurons that input into this neuron).
		 * @return The dot product of the input and weight vectors.
		 */
		public double activation_function() {
			if (thresholdNeuron) {
				return 1.0;
			}
			double activation = 0.0;
			for (int i = 0; i < inputs_weights.size(); i++) {
				activation += inputs_weights.get(i).input * inputs_weights.get(i).weight;
			}
			return activation;
		}

		/**
		 * The sigmoid function (as opposed to the step function, which is not differentiable and therefore is mathematically inconvenient to use) used for this
		 * neuron's output function. It maps the activation of this neuron to, roughly speaking, a 1 or 0 depending on whether or not the activation exceeds
		 * the threshold value.
		 * @param activation The activation to test.
		 * @return The output, which is always -1 if this neuron is the threshold neuron.
		 */
		public double output_function(double activation) {
			if (thresholdNeuron) {
				return -1.0;
			} else if (inputNeuron) {
				return activation;
			}
			return 1 / (1 + Math.exp(-activation));
		}

		/**
		 * Basic constructor.
		 */
		public Neuron() {
			synapses = new HashMap<>();
			inputs_weights = new ArrayList<>();
		}

		/**
		 * Constructor that indicates if this neuron is a threshold neuron, an input neuron, or possibly neither.
		 * @param isInputNeuron Whether or not this neuron is an input neuron. 
		 * @param isThresholdNeuron Whether or not this neuron is a threshold neuron.
		 */
		public Neuron(boolean isInputNeuron, boolean isThresholdNeuron) {
			synapses = new HashMap<>();
			inputs_weights = new ArrayList<>();
			inputNeuron = isInputNeuron;
			thresholdNeuron = isThresholdNeuron;
		}
	}

	ArrayList<ArrayList<Neuron>> layers; // A list of all the layers in this network and the neurons each layer contains. The first is the input layer, the 
										 // last is the output layer, and all in between are hidden layers.
	double defaultThreshold = 1.0; // The initial threshold value used by every neuron's output function.
	// Neuron thresholdNeuron;

	/**
	 * Constructor that builds a neural network with the number of layers specified by numLayers and the number of nodes in each layer specified by nodesPerLayer.
	 * Initializes weights to random numbers between -1 and 1.
	 * @param numLayers The number of layers in this network.
	 * @param nodesPerLayer A mapping from integers, which represent layers, to the number of nodes that layer should have.
	 */
	public NeuralNetwork(int numLayers, HashMap<Integer, Integer> nodesPerLayer) {
		if (nodesPerLayer.keySet().size() != numLayers) {
			throw new IllegalArgumentException("The nodesPerLayer mapping must map each layer exactly once.");
		}
		layers = new ArrayList<>();
		ArrayList<Neuron> inputLayer = new ArrayList<>();
		for (int i = 0; i < nodesPerLayer.get(0); i++) {
			inputLayer.add(new Neuron(true, false));
		}
		inputLayer.add(new Neuron(false, true));
		layers.add(inputLayer);
		for (int i = 1; i < numLayers; i++) {
			if (!nodesPerLayer.containsKey(i)) {
				throw new IllegalArgumentException("The nodesPerLayer mapping must map each layer exactly once.");
			}
			ArrayList<Neuron> currentLayer = new ArrayList<>();
			for (int j = 0; j < nodesPerLayer.get(i); j++) {
				currentLayer.add(new Neuron());
			}
			for (Neuron n : layers.get(i - 1)) {
				for (Neuron m : currentLayer) {
					if (n.thresholdNeuron) {
						n.synapses.put(m, defaultThreshold);
					} else {
						n.synapses.put(m, 2 * (Math.random() - 0.5));
					}
				}
			}
			if (i != numLayers - 1) {
				currentLayer.add(new Neuron(false, true));				
			}
			layers.add(currentLayer);
		}
	}

	/**
	 * Runs this network on the given inputs and returns the final output.
	 * @param inputs The list of inputs to input into the neurons in the input layer of the network.
	 * @return A list of all the outputs that were outputted by neurons in the output layer.
	 */
	public double[] run(double[] inputs) {
		if (inputs.length != layers.get(0).size() - 1) {
			throw new IllegalArgumentException(layers.get(0).size() - 1 + " inputs required.");
		}
		int i = 0;
		for (Neuron n : layers.get(0)) {
			if (n.thresholdNeuron) {
				continue;
			}
			ArrayList<Input> firstInput = new ArrayList<>();
			firstInput.add(new Input(inputs[i], 1.0));
			n.inputs_weights = firstInput;
			i++;
		}
		for (int layer = 0; layer < layers.size() - 1; layer++) {
			for (Neuron n : layers.get(layer)) {
				for (Neuron m : n.synapses.keySet()) {
					m.inputs_weights.add(new Input(n.output_function(n.activation_function()), n.synapses.get(m)));
				}
			}
		}
		double[] outputs = new double[layers.get(layers.size() - 1).size()];
		i = 0;
		for (Neuron output_neuron : layers.get(layers.size() - 1)) {
			outputs[i] = output_neuron.output_function(output_neuron.activation_function());
			i++;
		}
		return outputs;
	}

	/**
	 * Method used to implement the back-propagation supervised learning algorithm. Optimizes this network's weights to minimize the error in the output based on
	 * a given input. Networks are usually trained with a large number of training samples so the weights can be optimized to a high degree of precision and
	 * robustness. However, the training process is computationally inefficient and taxing.
	 * @param sample The training sample to use for training.
	 */
	public void train(TrainingSample sample) {
		double rate = 0.5;
		HashMap<Neuron, Double> outputs = new HashMap<>();
		double[] actualOutputs = run(sample.inputs, outputs);
		HashMap<Neuron, Double> expectedOutputs = new HashMap<>();
		int i = 0;
		for (Neuron n : layers.get(layers.size() - 1)) {
			expectedOutputs.put(n, sample.expectedOutput[i]);
			i++;
		}
		HashMap<Neuron[], Double> memoized = new HashMap<>();
		for (int layer = layers.size() - 2; layer >= 0; layer--) {
			for (Neuron n : layers.get(layer)) {
				if (n.thresholdNeuron) {
					for (Neuron m : n.synapses.keySet()) {
						n.synapses.put(m, n.synapses.get(m) + (rate * ));
					}
				} else {
					for (Neuron m : n.synapses.keySet()) {
						n.synapses.put(m, n.synapses.get(m) - (rate * derivatives(m, n, layers.size() - 2 - layer, memoized, outputs, expectedOutputs)));
					}
				}
			}
		}

	}

	/**
	 * The error function used to measure the error in this neural network's solution to inputs, compared to the expected output. This is the function
	 * to minimize.
	 * This error function must be used for all training samples, as the derivatives helper method relies on this specific error function; different error
	 * functions have different algorithms for computing deeper derivatives with respect to weights, and so would require that the derivatives function be
	 * modified.
	 * @param solution The solution outputted by this neural network.
	 * @return The error.
	 */
	public final double error(double[] solutions, TrainingSample t) {
		if (solutions.length != t.expectedOutput.length) {
			throw new IllegalArgumentException("Inputted solution does not have appropriate length; expected " + t.expectedOutput.length);
		}
		double total = 0.0;
		for (int i = 0; i < solutions.length; i++) {
			total += Math.pow((solutions[i] - t.expectedOutput[i]), 2);
		}
		return 0.5 * total;
	}

	/**
	 * Helper method.
	 * A memoized function that recursively computes the derivative of the error function with respect to the weight between fromNeuron and toNeuron, for use in
	 * the gradient descent algorithm used to update weights. The memoization data structure doesn't actually store the derivatives themselves, but rather the
	 * delta coefficients used to compute them, which is what's used in the recursive step of the function.
	 * @param toNeuron The neuron in the layer directly downstream of the weight being updated.
	 * @param fromNeuron The neuron in the layer directly upstream the weight being updated.
	 * @param memoized The data structure used for memoizing computation of the delta coefficients, to ease the computational burden.
	 * @param outputs The set of outputs in the output layer.
	 * @param expectedOutputs The set of desired or expected outputs, from the training sample.
	 * @return The derivative of the error function, which must be of the form E = 0.5 * sum over neuron n in output layer (output(n) - expected(n))^2.
	 */
	private double derivatives(Neuron toNeuron, Neuron fromNeuron, int distanceFromOutput, HashMap<Neuron[], Double> memoized, HashMap<Neuron, Double> outputs, HashMap<Neuron, Double> expected) {
		Neuron[] neuronPair = {toNeuron, fromNeuron};
		if (memoized.containsKey(neuronPair)) {
			return memoized.get(neuronPair);
		} else {
			if (distanceFromOutput == 0) {
				double delta = (outputs.get(toNeuron) - expected.get(toNeuron)) * (outputs.get(toNeuron)) * (1 - outputs.get(toNeuron));
				memoized.put(neuronPair, delta);
				return delta * outputs.get(fromNeuron);
			} else {
				double ret = 0.0;
				for (Neuron n : toNeuron.synapses.keySet()) {
					double delta = derivatives(n, toNeuron, distanceFromOutput - 1, memoized, outputs, expected) * toNeuron.synapses.get(n) * (outputs.get(toNeuron) * (1 - outputs.get(toNeuron)));
					memoized.put(neuronPair, delta);
					ret += delta * outputs.get(fromNeuron);
				}
				return ret;
			}
		}
	}

	// HELPER CODE BELOW THIS LINE.

	/**
	 * Class that represents inputs into neurons. Essentially bundles each input value with its associated weight, for ease of use in activation function.
	 */
	private static class Input {
		double input; // The input value to use.
		double weight; // The weight to scale the input value by.

		/**
		 * Basic constructor.
		 * @param input The input value.
		 * @param weight The weight.
		 */
		public Input(double input, double weight) {
			this.input = input;
			this.weight = weight;
		}
	}

	/**
	 * Simple class used to encode training samples, which are simply example inputs into this network and the expected output.
	 */
	public class TrainingSample {
		double[] inputs; // The example inputs.
		double[] expectedOutput; // The expected outputs for the above inputs.

		/**
		 * Basic constructor.
		 */
		public TrainingSample() {
			inputs = new double[layers.get(0).size() - 1];
			expectedOutput = new double[layers.get(layers.size() - 1).size()];
		}

		/**
		 * Constructor that creates a training sample with a given input and expected output.
		 * @param inputs The inputs.
		 * @param expected The expected outputs, corresponding to each input.
		 */
		public TrainingSample(double[] inputs, double[] expected) {
			this.inputs = inputs;
			this.expectedOutput = expected;
		}
	}

	/**
	 * Helper method.
	 * Copy of the above run function, but also stores all the outputs of each neuron in the inputted mapping, outputs.
	 */
	private double[] run(double[] inputs, HashMap<Neuron, Double> all_outputs) {
		if (inputs.length != layers.get(0).size() - 1) {
			throw new IllegalArgumentException(layers.get(0).size() - 1 + " inputs required.");
		}
		int i = 0;
		for (Neuron n : layers.get(0)) {
			if (n.thresholdNeuron) {
				continue;
			}
			ArrayList<Input> firstInput = new ArrayList<>();
			firstInput.add(new Input(inputs[i], 1.0));
			n.inputs_weights = firstInput;
			i++;
		}
		for (int layer = 0; layer < layers.size() - 1; layer++) {
			for (Neuron n : layers.get(layer)) {
				for (Neuron m : n.synapses.keySet()) {
					double output = n.output_function(n.activation_function());
					m.inputs_weights.add(new Input(output, n.synapses.get(m)));
					all_outputs.put(n, output);
				}
			}
		}
		double[] outputs = new double[layers.get(layers.size() - 1).size()];
		i = 0;
		for (Neuron output_neuron : layers.get(layers.size() - 1)) {
			outputs[i] = output_neuron.output_function(output_neuron.activation_function());
			all_outputs.put(output_neuron, outputs[i]);
			i++;
		}
		return outputs;
	}

	/**
	 * Helper method.
	 * Prints all the weights in this network.
	 */
	public void printWeights() {
		int i = 0;
		for (ArrayList<Neuron> layer : layers) {
			if (i == 0) {
				System.out.println("INPUT LAYER:");
			} else if (i == layers.size() - 1) {
				System.out.println("OUTPUT LAYER:");
			} else {
				System.out.println("HIDDEN LAYER " + (i - 1) + ":");
			}
			int j = 0;
			for (Neuron n : layer) {
				if (n.thresholdNeuron) {
					System.out.println("\tTHRESHOLD NEURON - " + n);
				} else {
					System.out.println("\tNEURON " + j + " - " + n);
				}
				System.out.print("\t\t");
				for (Neuron m : n.synapses.keySet()) {
					System.out.print("(" + n.synapses.get(m) + ", " + m + ")" + ", ");
				}
				System.out.println();
				j++;
			}
			System.out.println();
			i++;
		}
	}
}