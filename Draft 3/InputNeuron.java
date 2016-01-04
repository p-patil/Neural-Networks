import java.util.ArrayList;

public class InputNeuron extends Neuron {
	int id;

	public InputNeuron(int identity) {
		super();
		id = identity;
	}

	@Override
	public double output_function(ArrayList<Double> inputVector) {
		this.output = inputVector.get(id);
		return this.output;
	}

	@Override
	public void updateWeights(double learning_rate) {
		for (Synapse s : this.downstreamSynapses) {
			s.out.updateWeights(learning_rate);
		}
	}

	@Override
	public double delta_coefficient(double expectedOutput) {
		for (Synapse s : this.downstreamSynapses) {
			s.out.delta_coefficient(expectedOutput);
		}
		return 0.0;
	}

	@Override
	public void clearCache() {
		output = Double.NaN;
	}
}