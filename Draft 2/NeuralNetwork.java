import java.lang.Math;
import java.util.HashMap;

/**
 * Basic implementation of a feed-forward neural network that uses the back-propagation algorithm with supervised learning. The implementation is generalized and 
 * handles arbitrary numbers of neurons, layers, training sets, etc.
 */
public class NeuralNetwork {
	HashMap<Neuron, HashMap<Neuron, Double>> weights; // A mapping from one neuron to another neuron in the next layer storing the weight between the neurons.
	// HashMap<Neuron, Double> all_outputs; // Mapping that keeps track of what every neuron outputted.
	InputNeuron[] input_layer; // The input layer.
	HiddenNeuron[][] hidden_layers; // The list of hidden layers.
	OutputNeuron[] output_layer; // The output layer.
	HashMap<Neuron[], BiasNeuron> biases; // A mapping from a layer of neurons to the neuron used for the bias element for every neuron in that layer.
	// double initialBias = 0.0; // The initial bias to use for every neuron.
	double learning_rate = 0.9; // The rate of learning in the backpropagation algorithm, ie the coefficient of the partial derivative of the error function in
								// the gradient descent method for updating weights.

	/**
	 * Constructor.
	 * Given the number of layers and the number of nodes per layer, constructs an artificial neural network with the appropriate dimensions and with randomized
	 * weights.
	 * @param numLayers The total number of layers in the network, including the necessary input and output layers.
	 * @param nodesPerLayer A mapping from layers to integers defining how many neurons are in each layer.
	 */
	public NeuralNetwork(int numLayers, HashMap<Integer, Integer> nodesPerLayer) {
		if (numLayers < 2) {
			throw new IllegalArgumentException("numLayers must be at least 2, one input and one output layer are required");
		}
		if (numLayers != nodesPerLayer.size()) {
			throw new IllegalArgumentException("numLayers must match the number of layers in nodesPerLayer");
		}
		for (int i = 0; i < nodesPerLayer.size(); i++) {
			if (!nodesPerLayer.containsKey(i)) {
				throw new IllegalArgumentException("nodesPerLayer must contain every number between 0 and " + (nodesPerLayer.size() - 1) + " as a key");
			}
		}
		weights = new HashMap<>();
		biases = new HashMap<>();
		// all_outputs = new HashMap<>();
		input_layer = new InputNeuron[nodesPerLayer.get(0)];
		for (int i = 0; i < input_layer.length; i++) {
			input_layer[i] = new InputNeuron();
		}
		if (numLayers > 2) {
			hidden_layers = new HiddenNeuron[numLayers - 2][];
			for (int i = 0; i < hidden_layers.length; i++) {
				hidden_layers[i] = new HiddenNeuron[nodesPerLayer.get(i + 1)];
				for (int j = 0; j < hidden_layers[i].length; j++) {
					hidden_layers[i][j] = new HiddenNeuron();
				}
			}
		}
		output_layer = new OutputNeuron[nodesPerLayer.get(nodesPerLayer.size() - 1)];
		for (int i = 0; i < output_layer.length; i++) {
			output_layer[i] = new OutputNeuron();
		}
		if (numLayers > 2) {
			for (Neuron n : input_layer) {
				HashMap<Neuron, Double> temp = new HashMap<>();
				for (Neuron m : hidden_layers[0]) {
					temp.put(m, 2 * (Math.random() - 0.5));
					weights.put(n, temp);
				}
			}
			for (int i = 0; i < hidden_layers.length - 1; i++) {
				for (Neuron n : hidden_layers[i]) {
					HashMap<Neuron, Double> temp = new HashMap<>();
					for (Neuron m : hidden_layers[i + 1]) {
						temp.put(m, 2 * (Math.random() - 0.5));
						weights.put(n, temp);
					}
				}
			}
			for (Neuron n : hidden_layers[hidden_layers.length - 1]) {
				HashMap<Neuron, Double> temp = new HashMap<>();
				for (Neuron m : output_layer) {
					temp.put(m, 2 * (Math.random() - 0.5));
					weights.put(n, temp);
				}
			}
			for (HiddenNeuron[] h_layer : hidden_layers) {
				biases.put(h_layer, new BiasNeuron());
			}
			biases.put(output_layer, new BiasNeuron());
			for (HiddenNeuron[] h_layer : hidden_layers) {
				HashMap<Neuron, Double> temp = new HashMap<>();
				for (HiddenNeuron n : h_layer) {
					// temp.put(n, initialBias);
					temp.put(n, Math.random() - 0.5);
				}
				weights.put(biases.get(h_layer), temp);
			}
			HashMap<Neuron, Double> temp = new HashMap<>();
			for (OutputNeuron n : output_layer) {
				// temp.put(n, initialBias);
				temp.put(n, Math.random() - 0.5);
			}
			weights.put(biases.get(output_layer), temp);
		} else {
			for (Neuron n : input_layer) {
				HashMap<Neuron, Double> temp = new HashMap<>();
				for (Neuron m : output_layer) {
					temp.put(m, 2 * (Math.random() - 0.5));
					weights.put(n, temp);
				}
			}
			biases.put(output_layer, new BiasNeuron());
			HashMap<Neuron, Double> temp = new HashMap<>();
			for (OutputNeuron n : output_layer) {
				// temp.put(n, initialBias);
				temp.put(n, Math.random() - 0.5);
			}
			weights.put(biases.get(output_layer), temp);
		}
	}

	/**
	 * Feeds a list of inputs through the network and runs the information through to the output.
	 * @param inputs The list of inputs to use.
	 * @return A list of the outputs produced, each corresponding to an output neuron.
	 */
	public double[] feedforward(double[] inputs) {
		if (inputs.length != input_layer.length) {
			throw new IllegalArgumentException(input_layer.length + " inputs required");
		}
		for (int i = 0; i < input_layer.length; i++) {
			input_layer[i].input = inputs[i];
		}
		if (hidden_layers != null) {
			for (HiddenNeuron n : hidden_layers[0]) {
				double[] neuron_input = new double[input_layer.length + 1];
				for (int i = 0; i < input_layer.length; i++) {
					// all_outputs.put(input_layer[i], input_layer[i].output_function());
					// neuron_input[i] = all_outputs.get(input_layer[i]) * weights.get(input_layer[i]).get(n);
					neuron_input[i] = input_layer[i].output_function() * weights.get(input_layer[i]).get(n);
				}
				// all_outputs.put(biases.get(hidden_layers[0]), biases.get(hidden_layers[0]).output_function());
				// neuron_input[neuron_input.length - 1] = all_outputs.get(biases.get(hidden_layers[0])) * weights.get(biases.get(hidden_layers[0])).get(n);
				neuron_input[neuron_input.length - 1] = biases.get(hidden_layers[0]).output_function() * weights.get(biases.get(hidden_layers[0])).get(n);
				n.input_weight_product = neuron_input;
			}
			for (int i = 1; i < hidden_layers.length; i++) {
				for (HiddenNeuron n : hidden_layers[i]) {
					double[] neuron_input = new double[hidden_layers[i - 1].length + 1];
					for (int j = 0; j < hidden_layers[i - 1].length; j++) {
						// all_outputs.put(hidden_layers[i - 1][j], hidden_layers[i - 1][j].output_function());
						// neuron_input[j] = all_outputs.get(hidden_layers[i - 1][j]) * weights.get(hidden_layers[i - 1][j]).get(n);
						neuron_input[j] = hidden_layers[i - 1][j].output_function() * weights.get(hidden_layers[i - 1][j]).get(n);
					}
					// all_outputs.put(biases.get(hidden_layers[i]), biases.get(hidden_layers[i]).output_function());
					// neuron_input[neuron_input.length - 1] = all_outputs.get(biases.get(hidden_layers[i])) * weights.get(biases.get(hidden_layers[i])).get(n);
					neuron_input[neuron_input.length - 1] = biases.get(hidden_layers[i]).output_function() * weights.get(biases.get(hidden_layers[i])).get(n);
					n.input_weight_product = neuron_input;
				}
			}
			for (OutputNeuron n : output_layer) {
				double[] neuron_input = new double[hidden_layers[hidden_layers.length - 1].length + 1];
				for (int i = 0; i < hidden_layers[hidden_layers.length - 1].length; i++) {
					// all_outputs.put(hidden_layers[hidden_layers.length - 1][i], hidden_layers[hidden_layers.length - 1][i].output_function());
					// neuron_input[i] = all_outputs.get(hidden_layers[hidden_layers.length - 1][i]) * weights.get(hidden_layers[hidden_layers.length - 1][i]).get(n);
					neuron_input[i] = hidden_layers[hidden_layers.length - 1][i].output_function() * weights.get(hidden_layers[hidden_layers.length - 1][i]).get(n);
				}
				// all_outputs.put(biases.get(output_layer), biases.get(output_layer).output_function());
				// neuron_input[neuron_input.length - 1] = all_outputs.get(biases.get(output_layer)) * weights.get(biases.get(output_layer)).get(n);
				neuron_input[neuron_input.length - 1] = biases.get(output_layer).output_function() * weights.get(biases.get(output_layer)).get(n);
				n.input_weight_product = neuron_input;
			}
		} else {
			for (OutputNeuron n : output_layer) {
				double[] neuron_input = new double[input_layer.length + 1];
				for (int i = 0; i < input_layer.length; i++) {
					// all_outputs.put(input_layer[i], input_layer[i].output_function());
					// neuron_input[i] = all_outputs.get(input_layer[i]) * weights.get(input_layer[i]).get(n);
					neuron_input[i] = input_layer[i].output_function() * weights.get(input_layer[i]).get(n);
				}
				// all_outputs.put(biases.get(output_layer), biases.get(output_layer).output_function());
				// neuron_input[neuron_input.length - 1] = all_outputs.get(biases.get(output_layer)) * weights.get(biases.get(output_layer)).get(n);
				neuron_input[neuron_input.length - 1] = biases.get(output_layer).output_function() * weights.get(biases.get(output_layer)).get(n);
				n.input_weight_product = neuron_input;
			}
		}
		double[] ret = new double[output_layer.length];
		for (int i = 0; i < output_layer.length; i++) {
			// all_outputs.put(output_layer[i], output_layer[i].output_function());
			// ret[i] = all_outputs.get(output_layer[i]);
			ret[i] = output_layer[i].output_function();
		}
		return ret;
	}

	/**
	 * Given a sample list of inputs and a list of expected outputs, slightly modifies the weights in this network to decrease the error in the actual output and 
	 * expected output. The techniques used are based on the backpropagation supervised learning algorithm and assume that a sigmoid activation function is used.
	 */
	public void backpropagate(double[] inputs, double[] expected) {
		if (expected.length != output_layer.length) {
			throw new IllegalArgumentException("Must expect as many outputs as there are output neurons, output_layer and expected must have same length");
		}
		// First, run the inputs through the network.
		feedforward(inputs);

		// Backwards propagate the errors throughout the network, constructing the list of delta coefficients along the way.
		HashMap<Neuron, Double> errors = new HashMap<>();
		for (int i = 0; i < output_layer.length; i++) {
			// errors.put(output_layer[i], expected[i] - all_outputs.get(output_layer[i]));
			errors.put(output_layer[i], expected[i] - output_layer[i].output);
		}
		for (HiddenNeuron n : hidden_layers[hidden_layers.length - 1]) {
			double error = 0.0;
			for (OutputNeuron m : output_layer) {
				error += weights.get(n).get(m) * errors.get(m);
			}
			errors.put(n, error);
		}
		for (int layer = hidden_layers.length - 2; layer >= 0; layer--) {
			for (HiddenNeuron n : hidden_layers[layer]) {
				double error = 0.0;
				for (HiddenNeuron m : hidden_layers[layer + 1]) {
					error += weights.get(n).get(m) * errors.get(m);
				}
				errors.put(n, error);
			}
		}
		// Update all the weights using the delta coefficient, the learning rate, and the remaining terms in the derivative (of the error function).
		for (HiddenNeuron n : hidden_layers[0]) {
			for (InputNeuron m : input_layer) {
				// double input_m = all_outputs.get(m) * weights.get(m).get(n);
				// double new_weight = (learning_rate) * all_outputs.get(n) * (1 - all_outputs.get(n)) * errors.get(n) * input_m;
				double input_m = m.output * weights.get(m).get(n);
				double new_weight = (learning_rate) * n.output * (1 - n.output) * errors.get(n) * input_m;
				new_weight += weights.get(m).get(n);
				weights.get(m).put(n, new_weight);
			}
		}
		for (int layer = 1; layer < hidden_layers.length; layer++) {
			for (HiddenNeuron n : hidden_layers[layer]) {
				for (HiddenNeuron m : hidden_layers[layer - 1]) {
					// double input_m = all_outputs.get(m) * weights.get(m).get(n);
					// double new_weight = (learning_rate) * all_outputs.get(n) * (1 - all_outputs.get(n)) * errors.get(n) * input_m;
					double input_m = m.output * weights.get(m).get(n);
					double new_weight = (learning_rate) * n.output * (1 - n.output) * errors.get(n) * input_m;
					new_weight += weights.get(m).get(n);
					weights.get(m).put(n, new_weight);
				}
			}
		}
		for (OutputNeuron n : output_layer) {
			for (HiddenNeuron m : hidden_layers[hidden_layers.length - 1]) {
				// double input_m = all_outputs.get(m) * weights.get(m).get(n);
				// double new_weight = (learning_rate) * all_outputs.get(n) * (1 - all_outputs.get(n)) * errors.get(n) * input_m;
				double input_m = m.output * weights.get(m).get(n);
				double new_weight = (learning_rate) * n.output * (1 - n.output) * errors.get(n) * input_m;
				new_weight += weights.get(m).get(n);
				weights.get(m).put(n, new_weight);
			}
		}
		// Update the weights of the bias neurons.
		updateBiasNeuronWeights(errors);
	}

	/**
	 * Given a set of training data, updates all the weights in the network to minimize the error in actual and expected outputs for every sample in the data set.
	 * Runs for a specified number of iterations, with more iterations corresponding to greater accuracy.
	 * @param all_inputs The list of inputs.
	 * @param all_expected The list of corresponding expected outputs for each input.
	 * @param max_iterations The maximum number of iterations to train the network on the data for.
	 */
	public void train(double[][] all_inputs, double[][] all_expected, int max_iterations) {
		if (all_inputs.length != all_expected.length) {
			throw new IllegalArgumentException("Set of inputs must correspond to the set of expected, all_inputs and all_expected must have same length");
		}
		for (int i = 0; i < max_iterations; i++) {
			for (int j = 0; j < all_inputs.length; j++) {
				backpropagate(all_inputs[j], all_expected[j]);
				for (OutputNeuron n : output_layer) {
					if (n.output == Double.NaN) {
						System.out.println("FUCK NaN - Iteration " + i);
						return;
					} else if (n.output == 0.0) {
						System.out.println("FUCK ZERO Iteration " + i);
						return;
					}
					System.out.println("OUTPUT: " + n.output);
				}
			}
		}
	}

	// HELPER METHODS BELOW THIS LINE.

	/**
	 * Prints out a human readable display of this neural network.
	 */
	public void print() {
		System.out.println("INPUT LAYER");
		if (hidden_layers != null) {
			for (int i = 0; i < input_layer.length; i++) {
				System.out.println("\tNEURON " + i + " -- " + input_layer[i]);
				for (HiddenNeuron n : hidden_layers[0]) {
					System.out.println("\t\t" + weights.get(input_layer[i]).get(n) + " -> " + n);
				}
			}
			for (int i = 0; i < hidden_layers.length - 1; i++) {
				System.out.println("HIDDEN LAYER " + i);
				for (int j = 0; j < hidden_layers[i].length; j++) {
					System.out.println("\tNEURON " + j + " -- " + hidden_layers[i][j]);
					for (HiddenNeuron n : hidden_layers[i + 1]) {
						System.out.println("\t\t" + weights.get(hidden_layers[i][j]).get(n) + " -> " + n);
					}
				}
				System.out.println("\tBIAS NEURON -- " + biases.get(hidden_layers[i]));
				for (Neuron n : weights.get(biases.get(hidden_layers[i])).keySet()) {
					System.out.println("\t\t" + weights.get(biases.get(hidden_layers[i])).get(n) + " -> " + n);
				}
			}
			System.out.println("HIDDEN LAYER " + (hidden_layers.length - 1));
			for (int i = 0; i < hidden_layers[hidden_layers.length - 1].length; i++) {
				System.out.println("\tNEURON " + i + " -- " + hidden_layers[hidden_layers.length - 1][i]);
				for (Neuron n : output_layer) {
					System.out.println("\t\t" + weights.get(hidden_layers[hidden_layers.length - 1][i]).get(n) + " -> " + n);
				}
			}
			System.out.println("\tBIAS NEURON -- " + biases.get(hidden_layers[hidden_layers.length - 1]));
			for (Neuron n : weights.get(biases.get(hidden_layers[hidden_layers.length - 1])).keySet()) {
				System.out.println("\t\t" + weights.get(biases.get(hidden_layers[hidden_layers.length - 1])).get(n) + " -> " + n);
			}
		} else {
			for (int i = 0; i < input_layer.length; i++) {
				System.out.println("\tNEURON " + i + " -- " + input_layer[i]);
				for (Neuron n : output_layer) {
					System.out.println("\t\t" + weights.get(input_layer[i]).get(n) + " -> " + n);					
				}
			}
		}
		System.out.println("OUTPUT LAYER");
		for (int i = 0; i < output_layer.length; i++) {
			System.out.println("\tNEURON " + i + " -- " + output_layer[i]);
			// System.out.println("\t\t--OUTPUT--> " + all_outputs.get(output_layer[i]));
			System.out.println("\t\t--OUTPUT--> " + output_layer[i].output);
		}
		System.out.println("\tBIAS NEURON -- " + biases.get(output_layer));
		for (Neuron n : weights.get(biases.get(output_layer)).keySet()) {
			System.out.println("\t\t" + weights.get(biases.get(output_layer)).get(n) + " -> " + n);
		}
	}

	public void printOutputs() {
		System.out.println("INPUT LAYER");
		if (hidden_layers != null) {
			for (int i = 0; i < input_layer.length; i++) {
				System.out.println("\tNEURON " + i + " -- " + input_layer[i] + ", OUTPUT: " + input_layer[i].output);
				for (HiddenNeuron n : hidden_layers[0]) {
					System.out.println("\t\t" + weights.get(input_layer[i]).get(n) + " -> " + n);
				}
			}
			for (int i = 0; i < hidden_layers.length - 1; i++) {
				System.out.println("HIDDEN LAYER " + i);
				for (int j = 0; j < hidden_layers[i].length; j++) {
					System.out.println("\tNEURON " + j + " -- " + hidden_layers[i][j] + ", OUTPUT " + hidden_layers[i][j].output);
					for (HiddenNeuron n : hidden_layers[i + 1]) {
						System.out.println("\t\t" + weights.get(hidden_layers[i][j]).get(n) + " -> " + n);
					}
				}
				System.out.println("\tBIAS NEURON -- " + biases.get(hidden_layers[i]));
				for (Neuron n : weights.get(biases.get(hidden_layers[i])).keySet()) {
					System.out.println("\t\t" + weights.get(biases.get(hidden_layers[i])).get(n) + " -> " + n);
				}
			}
			System.out.println("HIDDEN LAYER " + (hidden_layers.length - 1));
			for (int i = 0; i < hidden_layers[hidden_layers.length - 1].length; i++) {
				System.out.println("\tNEURON " + i + " -- " + hidden_layers[hidden_layers.length - 1][i] + ", OUTPUT: " + hidden_layers[hidden_layers.length - 1][i].output);
				for (Neuron n : output_layer) {
					System.out.println("\t\t" + weights.get(hidden_layers[hidden_layers.length - 1][i]).get(n) + " -> " + n);
				}
			}
			System.out.println("\tBIAS NEURON -- " + biases.get(hidden_layers[hidden_layers.length - 1]));
			for (Neuron n : weights.get(biases.get(hidden_layers[hidden_layers.length - 1])).keySet()) {
				System.out.println("\t\t" + weights.get(biases.get(hidden_layers[hidden_layers.length - 1])).get(n) + " -> " + n);
			}
		} else {
			for (int i = 0; i < input_layer.length; i++) {
				System.out.println("\tNEURON " + i + " -- " + input_layer[i] + "OUTPUT: " + input_layer[i].output);
				for (Neuron n : output_layer) {
					System.out.println("\t\t" + weights.get(input_layer[i]).get(n) + " -> " + n);					
				}
			}
		}
		System.out.println("OUTPUT LAYER");
		for (int i = 0; i < output_layer.length; i++) {
			System.out.println("\tNEURON " + i + " -- " + output_layer[i] + ", OUTPUT: " + output_layer[i].output);
			// System.out.println("\t\t--OUTPUT--> " + all_outputs.get(output_layer[i]));
			System.out.println("\t\t--OUTPUT--> " + output_layer[i].output);
		}
		System.out.println("\tBIAS NEURON -- " + biases.get(output_layer));
		for (Neuron n : weights.get(biases.get(output_layer)).keySet()) {
			System.out.println("\t\t" + weights.get(biases.get(output_layer)).get(n) + " -> " + n);
		}
	}

	/**
	 * Given a partially constructed mapping from neurons to their associated error signals in the backpropagation algorithm, adds the appropriate error signals
	 * for the bias neurons to the mapping.
	 * @param errors The mapping of neurons to their error signals.
	 */
	private void updateBiasNeuronWeights(HashMap<Neuron, Double> errors) {
		for (int layer = 0; layer < hidden_layers.length; layer++) {
			double error = 0.0;
			for (Neuron n : hidden_layers[layer]) {
				error += weights.get(biases.get(hidden_layers[layer])).get(n) * errors.get(n);
			}
			errors.put(biases.get(hidden_layers[layer]), error);
		}
		double error = 0.0;
		for (Neuron n : output_layer) {
			error += weights.get(biases.get(output_layer)).get(n) * errors.get(n);
		}
		errors.put(biases.get(output_layer), error);
		for (int layer = 0; layer < hidden_layers.length; layer++) {
			for (Neuron n : hidden_layers[layer]) {
				double new_weight = (learning_rate) * errors.get(biases.get(hidden_layers[layer]));
				new_weight += weights.get(biases.get(hidden_layers[layer])).get(n);
				weights.get(biases.get(hidden_layers[layer])).put(n, new_weight);
			}
		}
		for (Neuron n : output_layer) {
			double new_weight = (learning_rate) * errors.get(biases.get(output_layer));
			new_weight += weights.get(biases.get(output_layer)).get(n);
			weights.get(biases.get(output_layer)).put(n, new_weight);
		}
	}
}