public class InputNeuron extends Neuron {
	double input;

	public double output_function() {
		this.output = input;
		return input;
	}

	public InputNeuron() {

	}
}