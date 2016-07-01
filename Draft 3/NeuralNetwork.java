import java.util.HashMap;
import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork {
	ArrayList<InputNeuron> inputNeurons;
	Neuron outputNeurons;
	double learning_rate = 0.25;
	ArrayList<ArrayList<Neuron>> hidden_layers = null;

	public NeuralNetwork(HashMap<Integer, Integer> nodesPerLayer) {
		inputNeurons = new ArrayList<>();
		for (int i = 0; i < nodesPerLayer.get(0); i++) {
			inputNeurons.add(new InputNeuron(i));
		}
		outputNeurons = new Neuron();
		if (nodesPerLayer.size() > 2) {
			hidden_layers = new ArrayList<>();
			for (int i = 1; i < nodesPerLayer.size() - 1; i++) {
				ArrayList<Neuron> layer = new ArrayList<>();
				for (int j = 0; j < nodesPerLayer.get(i); j++) {
					layer.add(new Neuron());
				}
				hidden_layers.add(layer);
			}
			for (InputNeuron n : inputNeurons) {
				for (Neuron m : hidden_layers.get(0)) {
					Synapse s = new Synapse(n, m);
				}
			}
			for (int i = 0; i < hidden_layers.size() - 1; i++) {
				for (Neuron n : hidden_layers.get(i)) {
					for (Neuron m : hidden_layers.get(i + 1)) {
						Synapse s = new Synapse(n, m);
					}
				}
			}
			for (Neuron n : hidden_layers.get(hidden_layers.size() - 1)) {
				Synapse s = new Synapse(n, outputNeurons);
			}
		} else {
			for (Neuron n : inputNeurons) {
				Synapse s = new Synapse(n, outputNeurons);
			}
		}
	}

	public double feedforward(ArrayList<Double> inputVector) {
		if (inputVector.size() != inputNeurons.size()) {
			throw new IllegalArgumentException(inputNeurons.size() + " inputs required");
		}
		outputNeurons.clearCache();
		double output = outputNeurons.output_function(inputVector);
		return output;
	}

	public void backpropagate(ArrayList<Double> inputVector, double expectedOutputs, double l_rate) {
		feedforward(inputVector);
		for (InputNeuron n : inputNeurons) {
			n.delta_coefficient(expectedOutputs);
		}
		for (InputNeuron n : inputNeurons) {
			n.updateWeights(l_rate);
		}
	}

	public void train(double[][] inputs, double[] expectedOutputs, int max_iterations, double l_rate) {
		if (inputs.length != expectedOutputs.length) {
			throw new IllegalArgumentException("The number of inputs must equal the number of expected outputs");
		}
		for (int i = 0; i < max_iterations; i++) {
			for (int j = 0; j < inputs.length; j++) {
				ArrayList<Double> inputsAsList = new ArrayList<>();
				for (double d : inputs[j]) {
					inputsAsList.add(d);
				}
				backpropagate(inputsAsList, expectedOutputs[j], l_rate);
			}
		}
	}

	public void train(double[][] inputs, double[] expectedOutputs, int max_iterations) {
		if (inputs.length != expectedOutputs.length) {
			throw new IllegalArgumentException("The number of inputs must equal the number of expected outputs");
		}
		for (int i = 0; i < max_iterations; i++) {
			for (int j = 0; j < inputs.length; j++) {
				ArrayList<Double> inputsAsList = new ArrayList<>();
				for (double d : inputs[j]) {
					inputsAsList.add(d);
				}
				backpropagate(inputsAsList, expectedOutputs[j], learning_rate);
			}
		}
	}

	public void print() {
		System.out.println("INPUT LAYER");
		for (Neuron n : inputNeurons) {
			System.out.println("\t" + n);
			for (Synapse s : n.downstreamSynapses) {
				System.out.println("\t\t" + s.weight + " - " + s.out);
			}
		}
		for (int i = 0; i < hidden_layers.size(); i++) {
			System.out.println("HIDDEN LAYER " + i);
			for (Neuron n : hidden_layers.get(i)) {
				System.out.println("\t" + n);
				for (Synapse s : n.downstreamSynapses) {
					System.out.println("\t\t" + s.weight + " - " + s.out);
				}
				for (Synapse s : n.incomingSynapses) {
					if (s.in == n.bias) {
						System.out.println("\t\tBIAS: " + s.weight + " - " + n.bias);						
					}
				}
			}
		}
		System.out.println("OUTPUT LAYER");
		System.out.println("\t" + outputNeurons);
		System.out.println("\t\tOUTPUT: " + outputNeurons.output);
		for (Synapse s : outputNeurons.incomingSynapses) {
			if (s.in == outputNeurons.bias) {
				System.out.println("\t\tBIAS: " + s.weight + " - " + outputNeurons.bias);						
			}
		}
	}
}