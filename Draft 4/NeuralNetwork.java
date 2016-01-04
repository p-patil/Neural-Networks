import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;

/**
 * Basic implementation of a feed-forward neural network that uses the backpropagation of errors algorithm with supervised learning and implements the stochastic 
 * gradient descent algorithm, once per training sample, to update synaptic weights; thus, this algorithm uses online (ie incremental) learning. The implementation
 * is generalized and handles arbitrary numbers of neurons, layers, training sets, etc. Allows for the usage of modifications such as the momentum factor or an 
 * adaptive learning rate.
 * A consequence of using a sigmoid threshold function is that the neural network's performance near 0 and 1 is very poor, since the 
 * sigmoid function can only attain these values in the limit and so it would require impractically large weights to reach values near 0 or 1. Thus, when using 
 * the network in situations which expect values near 0 or 1, first linearly transform (usually a simple translation and perhaps a scaling to readjust the domain)
 * the function to get outputs for the translated function and then inverse transform back to the original function; this will work much better. Moreover, for
 * this very nature of sigmoid functions, training set labels should be normalized prior to training.
 */
public class NeuralNetwork {
	ArrayList<InputNeuron> inputLayer; // Input layer of neurons.
	ArrayList<OutputNeuron> outputLayer; // Output layer of neurons that outputs the final answers.
	ArrayList<ArrayList<Neuron>> layers; // List of all hidden layers; excludes input and output layers.

	/**
	 * Class for the basic processing units of the neural network, neurons.
	 */
	public class Neuron {
		double output; // This neuron's output, which is sent to the next layer.
		ArrayList<Double> inputs; // The list of inputs received from neurons in the previous layer.
		double delta; // This neuron's delta coefficient, for use in updating weights with the backpropagation algorithm.
		ArrayList<Synapse> downstreamSynapses; // The connections to neurons in the next layer.
		ArrayList<Synapse> upstreamSynapses; // The connections to neurons in the previous layer.
		double cache; // Whether or not the output variable currently holds a value, since if it does, it must be cleared before it can be replaced.
		double adaptiveLearningRate; // Variable that stores the previous, ie before this iteration, learning rate for use in adapting the current learning rate.

		/**
		 * Basic constructor that initializes all instance variables to appropriate initial values.
		 */
		public Neuron() {
			this.output = Double.NaN;
			this.inputs = null;
			this.delta = Double.NaN;
			this.downstreamSynapses = new ArrayList<>();
			this.upstreamSynapses = new ArrayList<>();
			if (!(this instanceof InputNeuron)) {
				this.upstreamSynapses.add(new Synapse(new ThresholdNeuron(), this));
			}
			this.adaptiveLearningRate = Math.random();
		}

		/**
		 * Runs a given inputVector through the entire network, starting at the input layer and terminating at this neuron.
		 * @param inputVector The set of inputs to run on.
		 * @return The output value of this neuron, or the final output if this neuron is in the output layer.
		 */
		public double feedforward(ArrayList<Double> inputVector) {
			if (!Double.isNaN(this.output)) {
				return this.output;
			}
			this.inputs = new ArrayList<>();
			double dot_product = 0.0;
			for (Synapse s : this.upstreamSynapses) {
				double input = s.in.feedforward(inputVector);
				inputs.add(input);
				dot_product += s.weight * input;
			}
			this.output = sigmoid(dot_product);
			cache = output;
			return this.output;
		}

		/**
		 * Recursively (with memoization for efficiency) calculates the delta coefficients for this neuron, using the delta coefficients of the neurons in the
		 * next layer. The delta coefficient is a critical and the dynamic component of the weight update mechanism.
		 * Stores the coefficient by setting this neuron's error variable to the delta coefficients.
		 * @param expected The set of expected outputs.
		 * @return The delta coefficient.
		 */
		protected double delta_coefficient(ArrayList<Double> expected) {
			if (!Double.isNaN(this.delta)) {
				return this.delta;
			}
			this.delta = 0.0;
			for (Synapse s : this.downstreamSynapses) {
				this.delta += s.weight * s.out.delta_coefficient(expected);
			}
			return this.delta;
		}

		/**
		 * Updates the weights between this neuron and the neurons of the next layer, using the delta coefficients. Also updates the weights (recursively) of
		 * the neurons in the previous layer.
		 * @param learning_rate The factor to multiply the derivative of the error function with respect to the weight by. Controls how quickly weights are
		 *		  updated towards their optimal value. Values too low induce very slow learning, and values too high may cause the weights to overshoot the
		 *		  target weight and get too big and unstable.
		 */
		protected void updateWeights(double learning_rate) {
			if (!Double.isNaN(this.delta) && !Double.isNaN(this.output) && this.inputs != null) {
				for (int i = 0; i < this.upstreamSynapses.size(); i++) {
					this.upstreamSynapses.get(i).previousWeight = this.upstreamSynapses.get(i).weight;
					this.upstreamSynapses.get(i).previousGradient = this.output * (1 - this.output) * this.delta * this.inputs.get(i);
					this.upstreamSynapses.get(i).weight += (-learning_rate) * this.upstreamSynapses.get(i).previousGradient;
				}
				for (Synapse s : this.downstreamSynapses) {
					s.out.updateWeights(learning_rate);
				}
				this.delta = Double.NaN;
				this.output = Double.NaN;
				this.inputs = null;
			}
		}

		/**
		 * Same as default updateWeights method, but uses an inputted momentum factor meant to increase speed of convergence and avoid local minima by encouraging
		 * and discouraging beneficial and negative weight updates, respectively.
		 */
		protected void updateWeights_withMomentum(double momentum, double learning_rate) {
			if (!Double.isNaN(this.delta) && !Double.isNaN(this.output) && this.inputs != null) {
				for (int i = 0; i < this.upstreamSynapses.size(); i++) {
					this.upstreamSynapses.get(i).previousWeight = this.upstreamSynapses.get(i).weight;
					double currGradient = this.output * (1 - this.output) * this.delta * this.inputs.get(i);
					this.upstreamSynapses.get(i).weight += (-learning_rate) * currGradient + (momentum) * (-learning_rate) * this.upstreamSynapses.get(i).previousGradient;
					this.upstreamSynapses.get(i).previousGradient = currGradient;
				}
				for (Synapse s : this.downstreamSynapses) {
					s.out.updateWeights_withMomentum(momentum, learning_rate);
				}
				this.delta = Double.NaN;
				this.output = Double.NaN;
				this.inputs = null;
			}
		}

		/**
		 * Same as default updateWeights method, but applied the method of gradient descent to the learning rate itself before using the learning rate to update
		 * the weights. The learning rate gradually converges to its optimal value in real time; its speed of convergence is based on the inputted meta learning
		 * rate. An exponential moving average of the gradients of the error function for each synapse for each epoch is used to update the learning rate.
		 * DOESNT WORK.
		 */
		protected void updateWeights_withAdaptiveLearning(double meta_learning_rate, double expAvgCoefficient) {
			if (expAvgCoefficient < 0 || expAvgCoefficient > 1) {
				throw new IllegalArgumentException("Coefficient of exponential moving average must be between 0 and 1");
			}
			if (!Double.isNaN(this.delta) && !Double.isNaN(this.output) && this.inputs != null) {
				for (int i = 0; i < this.upstreamSynapses.size(); i++) {
					this.upstreamSynapses.get(i).previousWeight = this.upstreamSynapses.get(i).weight;
					double currGradient = this.output * (1 - this.output) * this.delta * this.inputs.get(i);
					// this.adaptiveLearningRate += (-meta_learning_rate) * currGradient * this.upstreamSynapses.get(i).previousGradient;
					// this.adaptiveLearningRate *= Math.exp((-meta_learning_rate) * currGradient * this.upstreamSynapses.get(i).previousGradient);
					if (Double.isNaN(this.upstreamSynapses.get(i).squaredGradientExpAvg)) {
						this.upstreamSynapses.get(i).squaredGradientExpAvg = Math.pow(currGradient, 2);
					} else {
						this.upstreamSynapses.get(i).squaredGradientExpAvg = expAvgCoefficient * this.upstreamSynapses.get(i).squaredGradientExpAvg + (1 - expAvgCoefficient) * Math.pow(this.upstreamSynapses.get(i).previousGradient, 2);
					}
					this.adaptiveLearningRate *= Math.max(0.5, 1 + meta_learning_rate * currGradient * (this.upstreamSynapses.get(i).gradientExpAvg / this.upstreamSynapses.get(i).squaredGradientExpAvg));
					if (Double.isNaN(this.upstreamSynapses.get(i).gradientExpAvg)) {
						this.upstreamSynapses.get(i).gradientExpAvg = currGradient;
					} else {
						this.upstreamSynapses.get(i).gradientExpAvg = expAvgCoefficient * this.upstreamSynapses.get(i).gradientExpAvg + (1 - expAvgCoefficient) * this.upstreamSynapses.get(i).previousGradient;
					}
					// this.adaptiveLearningRate *= Math.exp((-meta_learning_rate) * currGradient * this.upstreamSynapses.get(i).gradientExpAvg);
					System.out.println((this.upstreamSynapses.get(i).gradientExpAvg / this.upstreamSynapses.get(i).squaredGradientExpAvg));
					this.upstreamSynapses.get(i).weight += (-this.adaptiveLearningRate) * currGradient;
					this.upstreamSynapses.get(i).previousGradient = currGradient;
				}
				for (Synapse s : this.downstreamSynapses) {
					s.out.updateWeights_withAdaptiveLearning(meta_learning_rate, expAvgCoefficient);
				}
				this.delta = Double.NaN;
				this.output = Double.NaN;
				this.inputs = null;
			}
		}

		/**
		 * Sigmoid threshold function used as a differentiable replacement for the threshold step function theoretically used as well as used by biological 
		 * neurons.
		 * @param x The input.
		 * @return The function's value.
		 */
		private double sigmoid(double x) {
			return 1.0 / (1.0 + Math.exp(-x));
		}

		/**
		 * Clears all the output variables for this neuron and every neuron upstream of this one, clearing the way for another round of weight updating to take
		 * place.
		 */
		protected void clearOutputs() {
			if (!Double.isNaN(this.output)) {
				this.output = Double.NaN;
				for (Synapse s : this.upstreamSynapses) {
					s.in.clearOutputs();
				}
			}
		}
	}

	/**
	 * Class representing a neuron in the input layer.
	 */
	public class InputNeuron extends Neuron {
		int id; // A unique identification number useful in identifying which input of an input vector corresponds to which input neuron.

		/**
		 * Basic constructor.
		 */
		public InputNeuron(int id) {
			super();
			this.id = id; 
		}

		/**
		 * Initializes the running of this network.
		 */
		@Override
		public double feedforward(ArrayList<Double> inputVector) {
			this.output = inputVector.get(this.id);
			return this.output;
		}

		/**
		 * Initializes the computation and storage of all the delta coefficients.
		 */
		@Override
		protected double delta_coefficient(ArrayList<Double> expected) {
			for (Synapse s : this.downstreamSynapses) {
				s.out.delta_coefficient(expected);
			}
			return 0.0;
		}

		/**
		 * Initializes the updating of weights.
		 */
		@Override
		protected void updateWeights(double learning_rate) {
			for (Synapse s : this.downstreamSynapses) {
				s.out.updateWeights(learning_rate);
			}
		}

		@Override
		protected void updateWeights_withMomentum(double momentum, double learning_rate) {
			for (Synapse s : this.downstreamSynapses) {
				s.out.updateWeights_withMomentum(momentum, learning_rate);
			}
		}

		@Override
		protected void updateWeights_withAdaptiveLearning(double meta_learning_rate, double expAvgCoefficient) {
			for (Synapse s : this.downstreamSynapses) {
				s.out.updateWeights_withAdaptiveLearning(meta_learning_rate, expAvgCoefficient);
			}
		}

		@Override
		protected void clearOutputs() {
			this.output = Double.NaN;
		}
	}

	/**
	 * In order for the threshold value used in the threshold function (ie sigmoid function) in each neuron to be updated, we instead set a constant default
	 * threshold of 0 and subtract off the real threshold value from the activation to achieve the same effect. We add a filler neuron, called a threshold
	 * neuron, to always output 1 and whose weight is the threshold; thus, when weights are updated in the backpropagation algorithm, so is the threshold.
	 * This class represents threshold neurons.
	 */
	public class ThresholdNeuron extends InputNeuron {
		/**
		 * Basic constructor.
		 */
		public ThresholdNeuron() {
			super(-1);
		}

		/**
		 * Always return 1, so only the weight can be updated.
		 */
		@Override
		public double feedforward(ArrayList<Double> inputVector) {
			return 1.0;
		}
	}

	/**
	 * Class representing neurons in the output layer.
	 */
	public class OutputNeuron extends Neuron {
		int id; // A unique identification number useful in identifying which input of an input vector corresponds to which input neuron.

		/** 
		 * Basic constructor.
		 */
		public OutputNeuron(int id) {
			super();
			this.id = id;
		}

		/**
		 * The delta coefficient for an output neuron is simply the difference between the expected output and actual output for this neuron.
		 */
		@Override
		protected double delta_coefficient(ArrayList<Double> expected) {
			if (!Double.isNaN(this.delta)) {
				return this.delta;
			}
			this.delta = this.output - expected.get(id);
			return this.delta;
		}
	}

	/**
	 * Class representing the connections between neurons.
	 */
	public class Synapse {
		double weight; // The initially random weight of this connection.
		Neuron in; // The neuron sending information down this synapse.
		Neuron out; // The neuron receiving information from this synapse.
		double previousWeight; // In a training session, stores the weight in the last epoch, ie the one before the current one. This used in adaptive learning rates.
		double previousGradient; // The term used to update the weight in the last iteration of the backpropagation algorithm, used in momentum adaption.
		double gradientExpAvg; // A variable that keeps track of a moving exponential average of all the gradient terms this synapse has had in this epoch of training.
		double squaredGradientExpAvg; // Keeps track of moving exponential average of squared gradients.
		
		/**
		 * Basic constructor.
		 */
		public Synapse(Neuron in, Neuron out) {
			this.in = in;
			this.out = out;
			in.downstreamSynapses.add(this);
			out.upstreamSynapses.add(this);
			this.weight = 2 * (Math.random() - 0.5);
			this.previousWeight = this.weight;
			gradientExpAvg = Double.NaN;
			squaredGradientExpAvg = Double.NaN;
		}
	}

	/**
	 * Basic constructor.
	 */
	public NeuralNetwork() {
		this.inputLayer = new ArrayList<>();
		this.outputLayer = new ArrayList<>();
		this.layers = new ArrayList<>();
	}

	/**
	 * Constructor that initializes this neural network, with random weights, using the inputed mapping which maps layers to the number of nodes they should have.
	 * @param nodesPerLayer Array mapping an integer representing a layer (0 for input, last integer is output) and the value representing the number of nodes
	 * 						in that layer.
	 */
	public NeuralNetwork(int[] nodesPerLayer) {
		this();
		for (int i = 0; i < nodesPerLayer[0]; i++) {
			this.inputLayer.add(new InputNeuron(i));
		}
		for (int h_layer = 1; h_layer < nodesPerLayer.length - 1; h_layer++) {
			ArrayList<Neuron> layer = new ArrayList<>();
			for (int j = 0; j < nodesPerLayer[h_layer]; j++) {
				layer.add(new Neuron());
			}
			this.layers.add(layer);
		}
		for (int i = 0; i < nodesPerLayer[nodesPerLayer.length - 1]; i++) {
			this.outputLayer.add(new OutputNeuron(i));
		}
		for (InputNeuron n : this.inputLayer) {
			for (Neuron m : this.layers.get(0)) {
				new Synapse(n, m);
			}
		}
		for (int layer = 0; layer < nodesPerLayer.length - 3; layer++) {
			for (Neuron n : this.layers.get(layer)) {
				for (Neuron m : this.layers.get(layer + 1)) {
					new Synapse(n, m);
				}
			}
		}
		for (Neuron n : this.layers.get(layers.size() - 1)) {
			for (Neuron m : this.outputLayer) {
				new Synapse(n, m);
			}
		}
	}

	/**
	 * Runs the network on the inputted inputs.
	 * @param inputVector The list of inputs to use when running through the network.
	 */
	public ArrayList<Double> feedforward(ArrayList<Double> inputVector) {
		if (inputVector.size() != inputLayer.size()) {
			throw new IllegalArgumentException(inputLayer.size() + " inputs required");
		}
		for (Neuron n : this.outputLayer) {
			n.clearOutputs();
		}
		ArrayList<Double> output = new ArrayList<>();
		for (Neuron n : this.outputLayer) {
			output.add(n.feedforward(inputVector));
		}
		return output;
	}

	/**
	 * Implements the backpropagation algorithm for updating weights. Uses the inputted training sample to update the weights at a rate specified by learning_rate.
	 * @param inputs The input vector to use.
	 * @param expected The expected outputs.
	 * @param learning_rate How fast the network should learn, and also (inversely) how cautious it should be when learning.
	 */
	public void backpropagate(double[] inputs, double expected[], double learning_rate) {
		ArrayList<Double> inputsAsList = new ArrayList<>();
		for (double input : inputs) {
			inputsAsList.add(input);
		}
		ArrayList<Double> expectedAsList = new ArrayList<Double>();
		for (double expectedOutput : expected) {
			expectedAsList.add(expectedOutput);
		}
		feedforward(inputsAsList);
		for (InputNeuron n : this.inputLayer) {
			n.delta_coefficient(expectedAsList);
		}
		for (InputNeuron n : this.inputLayer) {
			n.updateWeights(learning_rate);
		}
	}

	/**
	 * Continuously updates weights to match a list of training samples. Terminates after a specified number of iterations of weight updating (on the same
	 * training samples).
	 * @param inputs The list of input vectors to use.
	 * @param expected The list of expected outputs.
	 * @param learning_rate The learning rate.
	 * @param max_iterations How many iterations to run for.
	 */
	public void train(double[][] inputs, double[][] expected, double learning_rate, int max_iterations) {
		if (inputs.length != expected.length) {
			throw new IllegalArgumentException("Number of input vectors must match number of expected outputs");
		}
		for (int i = 0; i < max_iterations; i++) {
			ArrayList<Integer> shuffle = new ArrayList<>();
			for (int j = 0; j <inputs.length; j++) {
				shuffle.add(j);
			}
			Collections.shuffle(shuffle);
			double[][] tempInputs = new double[inputs.length][inputs[0].length];
			double[][] tempExpected = new double[expected.length][expected[0].length];
			int count = 0;
			for (int j : shuffle) {
				tempInputs[j] = inputs[count];
				tempExpected[j] = expected[count];
				count++;
			}
			inputs = tempInputs;
			expected = tempExpected;
			for (int j = 0; j < inputs.length; j++) {
				backpropagate(inputs[j], expected[j], learning_rate);
			}
		}
	}	

	// OPTIONAL MODIFICATIONS BELOW THIS LINE.

	/**
	 * Same as above method, but includes a momentum factor which can increase training time but decreases probability of settling in local minima.
	 * @param inputs The input vector to use.
	 * @param expected The expected outputs.
	 * @param momentum The momentum to use.
	 * @param learning_rate How fast the network should learn, and also (inversely) how cautious it should be when learning.
	 */
	public void backpropagateWithMomentum(double[] inputs, double[] expected, double momentum, double learning_rate) {
		ArrayList<Double> inputsAsList = new ArrayList<>();
		for (double input : inputs) {
			inputsAsList.add(input);
		}
		ArrayList<Double> expectedAsList = new ArrayList<Double>();
		for (double expectedOutput : expected) {
			expectedAsList.add(expectedOutput);
		}
		feedforward(inputsAsList);
		for (InputNeuron n : this.inputLayer) {
			n.delta_coefficient(expectedAsList);
		}
		for (InputNeuron n : this.inputLayer) {
			n.updateWeights_withMomentum(momentum, learning_rate);
		}
	}

	/**
	 * Same as above method, but implements adaptive learning techniques.
	 * @param inputs The input vector to use.
	 * @param expected The expected outputs.
	 */
	public void backpropagateWithAdaptiveLearning(double[] inputs, double expected[]) {
		ArrayList<Double> inputsAsList = new ArrayList<>();
		for (double input : inputs) {
			inputsAsList.add(input);
		}
		ArrayList<Double> expectedAsList = new ArrayList<Double>();
		for (double expectedOutput : expected) {
			expectedAsList.add(expectedOutput);
		}
		feedforward(inputsAsList);
		for (InputNeuron n : this.inputLayer) {
			n.delta_coefficient(expectedAsList);
		}
		for (InputNeuron n : this.inputLayer) {
			n.updateWeights_withAdaptiveLearning(0.1, 0.5);
		}
	}

	/**
	 * Same as the default train method, but keeps training until the average error over all training samples is within a specified error tolerance.
	 * @param inputs The list of input vectors to use.
	 * @param expected The list of expected outputs.
	 * @param tolerance The error tolerance imposed on the network.
	 */
	public void train(double[][] inputs, double[][] expected, double tolerance) {
		if (inputs.length != expected.length) {
			throw new IllegalArgumentException("Number of input vectors must match number of expected outputs");
		}
		double avg_error = 0.0;
		do {
			ArrayList<Double> errors = new ArrayList<>();
			for (int i = 0; i < inputs.length; i++) {
				ArrayList<Double> inputsAsList = new ArrayList<>();
				for (double d : inputs[i]) {
					inputsAsList.add(d);
				}
				ArrayList<Double> actual = feedforward(inputsAsList);
				double err = 0.0;
				for (int j = 0; j < expected[i].length; j++) {
					err += Math.pow(expected[i][j] - actual.get(j), 2);
				}
				errors.add(err);
			}
			avg_error = 0.0;
			for (double err : errors) {
				avg_error += err;
			}
			avg_error /= errors.size();
			train(inputs, expected, 0.1, 100);
		} while (avg_error > tolerance);
	}

	/**
	 * Same as default train method, but uses a momentum factor (a fraction of the previous weight update is added on) in the weight update mechanism.
	 * @param inputs The list of input vectors to use.
	 * @param expected The list of expected outputs.
	 * @param learning_rate The learning rate.
	 * @param max_iterations How many iterations to run for.
	 */
	public void trainWithMomentum(double[][] inputs, double[][] expected, int max_iterations) {
		if (inputs.length != expected.length) {
			throw new IllegalArgumentException("Number of input vectors must match number of expected outputs");
		}
		for (int i = 0; i < max_iterations; i++) {
			ArrayList<Integer> shuffle = new ArrayList<>();
			for (int j = 0; j <inputs.length; j++) {
				shuffle.add(j);
			}
			Collections.shuffle(shuffle);
			double[][] tempInputs = new double[inputs.length][inputs[0].length];
			double[][] tempExpected = new double[expected.length][expected[0].length];
			int count = 0;
			for (int j : shuffle) {
				tempInputs[j] = inputs[count];
				tempExpected[j] = expected[count];
				count++;
			}
			inputs = tempInputs;
			expected = tempExpected;
			for (int j = 0; j < inputs.length; j++) {
				backpropagateWithMomentum(inputs[j], expected[j], 0.9, 0.7);
			}
		}
	}

	/**
	 * Same as default train method, but allows uses learning rate adaption in conjunction. Uses the bold driver algorithm.
	 * Doesn't work; learning rate increases too quickly.
	 * @param inputs The list of input vectors to use.
	 * @param expected The list of expected outputs.
	 * @param learning_rate The learning rate.
	 * @param max_iterations How many iterations to run for.
	 */
	public void trainWithAdaptingLearning_BoldDriver(double[][] inputs, double[][] expected, int max_iterations) {
		if (inputs.length != expected.length) {
			throw new IllegalArgumentException("Number of input vectors must match number of expected outputs");
		}
		double learning_rate = 0.1;
		for (int i = 0; i < max_iterations; i++) {
			System.out.println(learning_rate);
			ArrayList<Integer> shuffle = new ArrayList<>();
			for (int j = 0; j <inputs.length; j++) {
				shuffle.add(j);
			}
			Collections.shuffle(shuffle);
			double[][] tempInputs = new double[inputs.length][inputs[0].length];
			double[][] tempExpected = new double[expected.length][expected[0].length];
			int count = 0;
			for (int j : shuffle) {
				tempInputs[j] = inputs[count];
				tempExpected[j] = expected[count];
				count++;
			}
			inputs = tempInputs;
			expected = tempExpected;
			ArrayList<ArrayList<Double>> actual = new ArrayList<>();
			for (int j = 0; j < inputs.length; j++) {
				ArrayList<Double> temp = new ArrayList<>();
				for (double d : inputs[j]) {
					temp.add(d);
				}
				actual.add(feedforward(temp));
			}
			double prevError = errorFunction(expected, actual);
			for (int j = 0; j < inputs.length; j++) {
				backpropagate(inputs[j], expected[j], learning_rate);
			}
			double updatedError = errorFunction(expected, actual);
			if (updatedError <= prevError) {
				learning_rate *= 1.01;
			} else if (updatedError - prevError > 0.000001) {
				learning_rate *= 0.5;
			}
		}
	}

	/**
	 * Uses the Barzilai-Borwein learning rate update mechanism, which increases the learning rate following decreases in overall error, and decreases it when 
	 * instances of overshooting the error minimum, regression away from the error minimum, and, sometimes, of cycling in a local extremum are detected. Rather
	 * than updating a global learning rate that applies to the entire backpropagation algorithm, local learning rates unique to a particular synapse are updated.
	 * To prevent the learning rate from unexpectedly undergoing massive growth, overshooting the minimum, a maximum growth factor is introduced.
	 * @param inputs The list of input vectors to use.
	 * @param expected The list of expected outputs.
	 * @param learning_rate The learning rate.
	 * @param max_iterations How many iterations to run for.
	 */
	public void trainWithAdaptingLocalLearning(double[][] inputs, double[][] expected, int max_iterations) {
		if (inputs.length != expected.length) {
			throw new IllegalArgumentException("Number of input vectors must match number of expected outputs");
		}
		for (int i = 0; i < max_iterations; i++) {
			ArrayList<Integer> shuffle = new ArrayList<>();
			for (int j = 0; j <inputs.length; j++) {
				shuffle.add(j);
			}
			Collections.shuffle(shuffle);
			double[][] tempInputs = new double[inputs.length][inputs[0].length];
			double[][] tempExpected = new double[expected.length][expected[0].length];
			int count = 0;
			for (int j : shuffle) {
				tempInputs[j] = inputs[count];
				tempExpected[j] = expected[count];
				count++;
			}
			inputs = tempInputs;
			expected = tempExpected;
			ArrayList<ArrayList<Double>> actual = new ArrayList<>();
			for (int j = 0; j < inputs.length; j++) {
				ArrayList<Double> temp = new ArrayList<>();
				for (double d : inputs[j]) {
					temp.add(d);
				}
				actual.add(feedforward(temp));
			}
			for (int j = 0; j < inputs.length; j++) {
				backpropagateWithAdaptiveLearning(inputs[j], expected[j]);
			}
		}
	}

	// HELPER METHODS BELOW THIS LINE.

	/**
	 * Function used to compute the error in this network's outputs.
	 * @param inputs The inputs.
	 * @param expected The expected outputs.
	 * @param actual The received outputs.
	 * @return The error.
	 */
	private double errorFunction(double[][] expected, ArrayList<ArrayList<Double>> actual) {
		double sum = 0.0;
		for (int i = 0; i < expected.length; i++) {
			double dot_product = 0.0;
			for (int j = 0; j < expected[i].length; j++) {
				dot_product += Math.pow(expected[i][j] - actual.get(i).get(j), 2);
			}
			sum += dot_product;
		}
		return 0.5 * sum;
	}

	/**
	 * Computes the average error in this neural network's outputs for given training set.
	 * @param inputs The list of input vectors to use.
	 * @apram expected The list of output vectors to use as labels for each training sample.
	 * @return The average error.
	 */
	public double averageError(double[][] inputs, double[][] expected) {
		if (inputs.length != expected.length) {
			throw new IllegalArgumentException("List of inputs must have same size as list of expected outputs");
		}
		double err = 0.0;
		for (int i = 0; i < inputs.length; i++) {
			err += error(inputs[i], expected[i]);
		}
		return err / inputs.length;
	}

	/**
	 * Returns the error in this neural network's output on the given training sample.
	 * @param inputs The inputs of the training sample.
	 * @param expected The expected output vector, to compare against when computing the error.
	 * @return The error, as the magnitude of the difference vector in the actual output and expected output vectors.
	 */
	public double error(double[] inputs, double[] expected) {
		ArrayList<Double> inputsAsList = new ArrayList<>();
		for (int i = 0; i < inputs.length; i++) {
			inputsAsList.add(inputs[i]);
		}
		ArrayList<Double> actual = feedforward(inputsAsList);
		if (actual.size() != expected.length) {
			throw new IllegalArgumentException("List of expected outputs does not match number of output nodes");
		}
		double err = 0.0;
		for (int i = 0; i < actual.size(); i++) {
			err += Math.pow(actual.get(i) - expected[i], 2);
		}
		return Math.sqrt(err);
	}

	/**
	 * Re-randomizes all the weights in this network, effectively clearing any trace of past training or learning.
	 */
	public void clear() {
		int[] nodesPerLayer = new int[this.layers.size() + 2];
		nodesPerLayer[0] = inputLayer.size();
		for (int i = 0; i < this.layers.size(); i++) {
			nodesPerLayer[i + 1] = this.layers.get(i).size();
		}
		nodesPerLayer[this.layers.size()] = outputLayer.size();
		new NeuralNetwork(nodesPerLayer);
	}
}