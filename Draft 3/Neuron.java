import java.lang.Math;
import java.util.ArrayList;

public class Neuron {
	ArrayList<Synapse> downstreamSynapses = new ArrayList<>();
	ArrayList<Synapse> incomingSynapses = new ArrayList<>();
	double output = Double.NaN;
	ArrayList<Double> inputs = null;
	double delta = Double.NaN;
	BiasNeuron bias;
	double cache;

	public Neuron() {
		if (!(this instanceof InputNeuron)) {
			bias = new BiasNeuron();
			Synapse s = new Synapse(bias, this);
		}
	}

	public double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	public double output_function(ArrayList<Double> inputVector) {
		if (!Double.isNaN(output)) {
			return output;
		}
		inputs = new ArrayList<>();
		double dot_product = 0.0;
		for (Synapse s : incomingSynapses) {
			double input = s.in.output_function(inputs);
			inputs.add(input);
			dot_product += input * s.weight;
		}
		output = sigmoid(dot_product);
		cache = output;
		return output;
	}

	public double delta_coefficient(double expectedOutput) {
		if (!Double.isNaN(delta)) {
			return delta;
		}
		delta = 0.0;
		if (downstreamSynapses.isEmpty()) {
			delta = expectedOutput - output;
		} else {
			for (Synapse s : downstreamSynapses) {
				delta += s.weight * s.out.delta_coefficient(expectedOutput);
			}
		}
		return delta;
	}

	public void updateWeights(double learning_rate) {
		if (!Double.isNaN(delta) && !Double.isNaN(output) && inputs != null) {
			for (int i = 0; i < incomingSynapses.size(); i++) {
				incomingSynapses.get(i).weight += (learning_rate) * delta * output * (1 - output) * inputs.get(i);
			}
			for (Synapse s : downstreamSynapses) {
				s.out.updateWeights(learning_rate);
			}
			delta = Double.NaN;
			output = Double.NaN;
			inputs = null;
		}
	}

	public void clearCache() {
		if (!Double.isNaN(output)) {
			output = Double.NaN;
			for (Synapse s : incomingSynapses) {
				s.in.clearCache();
			}
		}
	}
}