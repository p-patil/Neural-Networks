import java.util.ArrayList;
import java.util.Random;
import java.util.Collections;
import java.lang.Math;

public class NeuralNetwork {
	int num_layers; // Number of layers.
	int[] layers; // Stores how many neurons each layer has, with layers corresponding to indexes and the value stored at the index being the size of the layer.
	ArrayList<double[]> biases; // List of vectors containing the biases. The i-th vector in the list contains the biases for layer i. The j-th entry in the vector
							    // contains the bias for neuron j in layer i.
	ArrayList<double[][]> weights; // List of matrices that contain weights. The i-th matrix in the list contains the weights between layer i and the next layer.
								   // Each matrix has as many rows as the (i + 1)-th layer has neurons, and as many columns as the i-th layer has neurons. The entry
								   // in the j-th row and k-th column is the weight between neuron k in the next layer and neuron j in the i-th layer. 

	/**
	 * Given a list of layers and their respective sizes, constructs a neural network with random weights and biases. The weights and biases are randomly chosen
	 * from the standard Gaussian distribution.
	 * @param nodesPerLayer Array whose i-th value represents the size of layer i in this network.
	 */
	public NeuralNetwork(int[] nodesPerLayer) {
		this.num_layers = nodesPerLayer.length;
		this.layers = nodesPerLayer;
		this.biases = new ArrayList<>();
		this.weights = new ArrayList<>();
		Random r = new Random();
		// Initializes each vector in biases. The input layer has no biases.
		for (int i = 1; i < nodesPerLayer.length; i++) {
			double[] b = new double[nodesPerLayer[i]];
			for (int j = 0; j < b.length; j++) {
				b[j] = r.nextGaussian();
			}
			this.biases.add(b);
		}
		// Initializes the matrices in weights.
		for (int i = 0; i < nodesPerLayer.length - 1; i++) {
			// Each row represents a neuron in layer i + 1 and each column represents a neuron in layer i. The entry at that location is the weight.
			double[][] w = new double[nodesPerLayer[i + 1]][nodesPerLayer[i]];
			for (int j = 0; j < w.length; j++) {
				for (int k = 0; k < w[j].length; k++) {
					w[j][k] = r.nextGaussian();
				}
			}
		}
	}

	/**
	 * Given a list of inputs, feeds the inputs through this neural network and returns the output.
	 * @param inputs The list of inputs to use.
	 * @return The outputs.
	 */
	public double[] feedforward(double[] inputs) {
		// If a is the activation vector for the i-th layer, w is the weights matrix for layers i and i + 1, and b is the bias vector for the i-th layer, the
		// activation vector for layer i + 1 is given by a' = sigmoid(wa + b).
		double[] activation_vector = inputs;
		for (int i = 0; i < this.weights.size(); i++) {
			activation_vector = activation_function(Matrix.add(Matrix.multiply(this.weights.get(i), activation_vector), biases.get(i)));
		}
		return activation_vector;
	}

	/**
	 * Implements the stochastic gradient descent algorithm to update all the weights and biases in this neural network, given a list of training samples. Uses
	 * the backpropagation algorithm. This is the method used when training this neural network.
	 * @param inputs The inputs to train with.
	 * @param outputs The corresponding expected outputs for each input.
	 * @param num_epochs How many iterations to train for.
	 * @param batch_size The size of the batches used when computing the gradient of the error function. A batch size of 1 is essentially the online (ie 
	 *					 incremental) learning method.
	 * @param learning_rate The learning rate to use.
	 */
	public void SGD(double[][] inputs, double[][] outputs, int num_epochs, int batch_size, double learning_rate) {
		if (inputs.length != outputs.length) {
			throw new IllegalArgumentException("Number of inputs must equal number of expected outputs");
		}
		if (batch_size > inputs.length) {
			throw new IllegalArgumentException("Batch size cannot be greater than the size of the training data set");
		}
		if (learning_rate <= 0) {
			throw new IllegalArgumentException("Learning rate must be positive");
		}
		for (int epoch = 0; epoch < num_epochs; epoch++) {
			inputs = shuffle(inputs);
			outputs = shuffle(outputs);
			double[][][] input_batches = new double[(inputs.length / batch_size) + 1][batch_size][];
			double[][][] output_batches = new double[(outputs.length / batch_size) + 1][batch_size][];
			for (int i = 0, j = 0; i < input_batches.length && j < inputs.length; i++, j += batch_size) {
				for (int k = 0; k < batch_size; k++) {
					input_batches[i][k] = new double[inputs[j + k].length];
					for (int l = 0; l < inputs[j + k].length; l++) {
						input_batches[i][k][l] = inputs[j + k][l];
					}
				}
			}
			for (int i = 0; i < input_batches.length; i++) {
				update_weights(input_batches[i], output_batches[i], learning_rate);
			}
			System.out.println("Iteration " + epoch + " completed.");
		}
	}

	/**
	 * Modifies original SGD method by accepting a test data set upon which to test the neural network's progress every epoch. The progress the neural network
	 * has made during a given epoch, defined as the proportion of cases with correct outputs, relative to expected test outputs, are printed every epoch.
	 * @param inputs The inputs to train with.
	 * @param outputs The corresponding expected outputs for each input.
	 * @param num_epochs How many iterations to train for.
	 * @param batch_size The size of the batches used when computing the gradient of the error function. A batch size of 1 is essentially the online (ie 
	 *					 incremental) learning method.
	 * @param learning_rate The learning rate to use.
	 * @param test_input List of inputs to try in the test data set.
	 * @param test_output List of expected outputs corresponding to the inputs in the test data set.
	 */ 
	public void SGD(double[][] inputs, double[][] outputs, int num_epochs, int batch_size, double learning_rate, double[][] test_input, double[][] test_output) {
		if (inputs.length != outputs.length) {
			throw new IllegalArgumentException("Number of inputs must equal number of expected outputs");
		}
		if (test_input.length != test_output.length) {
			throw new IllegalArgumentException("Number of test inputs must equal number of test outputs");
		}
		if (batch_size > inputs.length) {
			throw new IllegalArgumentException("Batch size cannot be greater than the size of the training data set");
		}
		if (learning_rate <= 0) {
			throw new IllegalArgumentException("Learning rate must be positive");
		}
		System.out.println("Accuracy per epoch:");
		for (int epoch = 0; epoch < num_epochs; epoch++) {
			inputs = shuffle(inputs);
			outputs = shuffle(outputs);
			double[][][] input_batches = new double[(inputs.length / batch_size) + 1][batch_size][];
			double[][][] output_batches = new double[(outputs.length / batch_size) + 1][batch_size][];
			for (int i = 0, j = 0; i < input_batches.length && j < inputs.length; i++, j += batch_size) {
				for (int k = 0; k < batch_size; k++) {
					input_batches[i][k] = new double[inputs[j + k].length];
					for (int l = 0; l < inputs[j + k].length; l++) {
						input_batches[i][k][l] = inputs[j + k][l];
					}
				}
			}
			for (int i = 0; i < input_batches.length; i++) {
				update_weights(input_batches[i], output_batches[i], learning_rate);
			}
			System.out.println("Iteration " + epoch + ": " + evaluate(test_input, test_output));
		}
	}

	/**
	 * Given a batch, ie a small sample of the training data set used during stochastic gradient descent, uses the backpropagation algorithm to update the weights
	 * and biases to their new values.
	 * @param input_batch The list of inputs in the batch.
	 * @param output_batch The list of corresponding expected outputs in the batch.
	 * @param learning_rate The learning rate to use; controls how big the steps made towards the minimum by the stochastic gradient descent algorithm are.
	 */
	private void update_weights(double[][] input_batch, double[][] output_batch, double learning_rate) {
		if (input_batch.length != output_batch.length) {
			System.out.println("Error in update: length of input_batch is not same as length of output_batch");
		}
		ArrayList<double[]> bias_gradient = new ArrayList<>();
		for (double[] bias_vector : this.biases) {
			bias_gradient.add(new double[bias_vector.length]);
		}
		ArrayList<double[][]> weight_gradient = new ArrayList<>();
		for (double[][] weight_matrix : weights) {
			weight_gradient.add(new double[weight_matrix.length][weight_matrix[0].length]);
		}
		for (int i = 0; i < input_batch.length; i++) {
			ArrayList<Object> backpropagated = backpropagate(input_batch[i], output_batch[i]);
			ArrayList<double[]> delta_bias = (ArrayList<double[]>) backpropagated.get(0);
			ArrayList<double[][]> delta_weight = (ArrayList<double[][]>) backpropagated.get(1);
			ArrayList<double[]> bias_gradient_copy = new ArrayList<>();
			for (int j = 0; j < bias_gradient.size(); j++) {
				bias_gradient_copy.add(Matrix.add(bias_gradient.get(j), delta_bias.get(j)));
			}
			ArrayList<double[][]> weight_gradient_copy = new ArrayList<>();
			for (int j = 0; j < weight_gradient.size(); j++) {
				weight_gradient_copy.add(Matrix.add(weight_gradient.get(j), delta_weight.get(j)));
			}
			bias_gradient = bias_gradient_copy;
			weight_gradient = weight_gradient_copy;
		}
		ArrayList<double[]> biases_copy = new ArrayList<>();
		ArrayList<double[][]> weights_copy = new ArrayList<>();
		for (int i = 0; i < this.biases.size(); i++) {
			biases_copy.add(Matrix.subtract(this.biases.get(i), (Matrix.multiply(learning_rate / input_batch.length, bias_gradient.get(i)))));
		}
		for (int i = 0; i < this.weights.size(); i++) {
			weights_copy.add(Matrix.subtract(this.weights.get(i), Matrix.multiply(learning_rate / input_batch.length, weight_gradient.get(i))));
		}
		this.biases = biases_copy;
		this.weights = weights_copy;
	}

	/**
	 * Direct implementation of standard backpropagation algorithm. Efficiently computes the gradient of the error function with respect to the weights and biases.
	 * @param inputs The input vector to use.
	 * @param outputs The expected output vector.
	 * @return A list of size two. The first element is a list of vectors, with the i-th vector being the gradient of error function with respect to the i-th bias
	 *		   vector in this.biases. The second element is a list of matrices, with the i-th matrix being the gradient of the error function with respect to the
	 *		   i-th weight matrix in this.weights.
	 */
	private ArrayList<Object> backpropagate(double[] inputs, double[] outputs) {
		// Initialize gradient vectors to return.
		ArrayList<double[]> bias_gradient = new ArrayList<>();
		for (double[] bias_vector : this.biases) {
			bias_gradient.add(new double[bias_vector.length]);
		}
		ArrayList<double[][]> weight_gradient = new ArrayList<>();
		for (double[][] weight_matrix : weights) {
			weight_gradient.add(new double[weight_matrix.length][weight_matrix[0].length]);
		}
		// Feedforward the input vector, computing storing all the neuron outputs along the way.
		ArrayList<double[]> neuron_outputs = new ArrayList<>(); // Stores the outputs of each layer prior to passing through the activation function.
		ArrayList<double[]> neuron_activations = new ArrayList<>(); // Stores the outputs of each layer after passing through the activation function.
		double[] neuron_activation = inputs;
		neuron_activations.add(neuron_activation);
		for (int i = 0; i < this.weights.size(); i++) {
			double[] neuron_output  = Matrix.add(Matrix.multiply(this.weights.get(i), neuron_activation), this.biases.get(i));
			neuron_outputs.add(neuron_output);
			neuron_activation = activation_function(neuron_output);
			neuron_activations.add(neuron_activation);
		}
		// Backwards propagation of errors.
		double[] delta_vector = Matrix.hadamardMultiply(dError_dActivationFunction(neuron_activations.get(neuron_activations.size() - 1), outputs), vectorized_sigmoid_derivative(neuron_outputs.get(neuron_outputs.size() - 1)));
		bias_gradient.set(this.num_layers - 1, delta_vector);
		// weight_gradient.set(this.num_layers - 1, Matrix.multiply(neuron_activations.get(neuron_activations.size() - 2), delta_vector));
		double[][] new_weight_matrix = new double[delta_vector.length][neuron_activations.get(neuron_activations.size() - 2).length];
		for (int i = 0; i < new_weight_matrix.length; i++) {
			for (int j = 0; j < new_weight_matrix[i].length; j++) {
				new_weight_matrix[i][j] = neuron_activations.get(neuron_activations.size() - 2)[j] * delta_vector[i];
			}
		}
		weight_gradient.set(this.num_layers - 1, new_weight_matrix);
		for (int i = this.num_layers - 2; i > 0; i--) {
			delta_vector = Matrix.hadamardMultiply(Matrix.multiply(this.weights.get(i + 1), delta_vector), activation_function_derivative(neuron_outputs.get(i)));
			bias_gradient.set(i, delta_vector);
			// weight_gradient.set(i, Matrix.multiply(neuron_activations.get(i - 1), delta_vector));
			new_weight_matrix = new double[delta_vector.length][neuron_activations.get(i - 1).length];
			for (int j = 0; j < new_weight_matrix.length; j++) {
				for (int k = 0; k < new_weight_matrix[j].length; k++) {
					new_weight_matrix[j][k] = neuron_activations.get(i - 1)[k] * delta_vector[j];
				}
			}
			weight_gradient.set(i, new_weight_matrix);
		}
		ArrayList<Object> ret = new ArrayList<>();
		ret.add(bias_gradient);
		ret.add(weight_gradient);
		return ret;
	}

	private double[] dError_dActivationFunction(double[] outputs, double[] expected) {
		// Assuming the error function is E = 1/2 * |expected - actual|^2, ie half the squared magnitude of the difference vector.
		return Matrix.subtract(expected, outputs);
	}

	/**
	 * The activation function to use. By default, uses the sigmoid function.
	 * @param matrix A matrix to apply the activation function on.
	 * @return The resulting matrix after applying the function.
	 */
	private double[][] activation_function(double[][] matrix) {
		return vectorized_sigmoid(matrix);
	}

	/**
	 * The activation function to use. By default, uses the sigmoid function.
	 * @param vector A vector to apply the activation function on.
	 * @return vector The resulting vector after applying the function.
	 */
	private double[] activation_function(double[] vector) {
		return vectorized_sigmoid(vector);
	}

	/**
	 * The derivative of the activation function. By default, uses the derivative of the sigmoid function.
	 * @param matrix The matrix on which to apply this function.
	 * @return The resulting matrix after applying the function.
	 */
	private double[][] activation_function_derivative(double[][] matrix) {
		return vectorized_sigmoid_derivative(matrix);
	}

	/**
	 * The derivative of the activation function. By default, uses the derivative of the sigmoid function.
	 * @param vector The vector on which to apply this function.
	 * @return The resulting vector after applying the function.
	 */
	private double[] activation_function_derivative(double[] vector) {
		return vectorized_sigmoid_derivative(vector);
	}

	/**
	 * Given a list of input vectors and corresponding expected output vectors, returns the proportion of cases in which this neural network was correct. Uses
	 * a helper method that checks two output vectors against each other assuming both represent outputs to a digit recognition problem
	 * @param test_input The list of input vectors to use.
	 * @param test_output The list of corresponding output vectors.
	 */
	private double evaluate(double[][] test_input, double[][] test_output) {
		int count = 0;
		for (int i = 0; i < test_input.length; i++) {
			if (digit_recognition_test(feedforward(test_input[i]), test_output[i])) {
				count++;
			}
		}
		return count / test_input.length;
	}

	// HELPER METHODS BELOW THIS LINE.

	/**
	 * Sigmoid function used by each neuron.
	 * @param x The argument.
	 * @return 1/(1 + e^(-x)).
	 */
	private static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	/**
	 * Evaluates the derivative of the sigmoid function.
	 * @param x The argument.
	 * @return s(x) * (1 - s(x)), where s = 1/(1 + e^(-x)).
	 */
	private static double sigmoid_derivative(double x) {
		double y = sigmoid(x);
		return y * (1.0 - y);
	}

	/**
	 * Vectorizes the sigmoid function for use on matrices; applies the sigmoid function to every entry in the matrix.
	 * @param matrix The matrix whose entries the sigmoid function should accept as input.
	 * @return The new matrix with entries in matrix but passed through the sigmoid function.
	 */
	private static double[][] vectorized_sigmoid(double[][] matrix) {
		double[][] ret = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			ret[i] = new double[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				ret[i][j] = sigmoid(matrix[i][j]);
			}
		}
		return ret;
	}

	/**
	 * Vectorizes the sigmoid function for use on vectors; applies the sigmoid function to every entry in the vector.
	 * @param vector The vector whose entries the sigmoid function should accept as input.
	 * @return The new vector with entries in vector but passed through the sigmoid function.
	 */
	private static double[] vectorized_sigmoid(double[] vector) {
		double[] ret = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			ret[i] = sigmoid(vector[i]);
		}
		return ret;
	}

	/**
	 * Applies the sigmoid_derivative function to every entry in matrix.
	 * @param matrix The matrix to apply the function on.
	 * @return The resulting matrix after applying the function.
	 */
	private static double[][] vectorized_sigmoid_derivative(double[][] matrix) {
		double[][] ret = new double[matrix.length][matrix[0].length];
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0;j < matrix[i].length; j++) {
				ret[i][j] = sigmoid_derivative(matrix[i][j]);
			}
		}
		return ret;
	}

	/**
	 * Applies the sigmoid_derivative function to every entry in vector.
	 * @param vector The vector to apply the function on.
	 * @return The resulting vector after applying the function.
	 */
	private static double[] vectorized_sigmoid_derivative(double[] vector) {
		double[] ret = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			ret[i] = sigmoid_derivative(vector[i]);
		}
		return ret;
	}

	/**
	 * Takes an array of arrays and shuffles the outer array's elements (ie the inner arrays) randomly, then returns the new array of arrays.
	 * @param arr The 2-D array to shuffle.
	 * @return A 2-D array with shuffled elements.
	 */
	private static double[][] shuffle(double[][] arr) {
		ArrayList<Integer> shuffle = new ArrayList<>();
		for (int j = 0; j <arr.length; j++) {
			shuffle.add(j);
		}
		Collections.shuffle(shuffle);
		double[][] tempArr = new double[arr.length][arr[0].length];
		int count = 0;
		for (int j : shuffle) {
			tempArr[j] = arr[count];
			count++;
		}
		return tempArr;
	}

	/**
	 * Compares two output vectors to see if they are the same in the context of digit recognition. The actual output vector's index at its maximum value is the
	 * digit guessed, and the expected output vector is filled with 0's and has a 1 at the index of the correct index. Returns if the two indexes are the same.
	 * @param actual The output vector outputted by a neural network when analyzing handwritten digits.
	 * @param expected The expected output vector.
	 * @return Whether or not the maximum indexes are the same.
	 */
	private static boolean digit_recognition_test(double[] actual, double[] expected) {
		int max_index = 0;
		for (int j = 0;j < actual.length; j++) {
			if (actual[j] > actual[max_index]) {
				max_index = j;
			}
		}
		int max_expected_index = 0;
		for (int j = 0; j < expected.length; j++) {
			if (expected[j] == 1) {
				max_expected_index = j;
				break;
			}
		}
		if (max_index == max_expected_index) {
			return true;
		}
		return false;
	}
}