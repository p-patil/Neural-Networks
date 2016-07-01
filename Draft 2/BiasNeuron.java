public class BiasNeuron extends Neuron {

	public double output_function() {
		return this.output;
	}

	public BiasNeuron() {
		this.output = 1.0;
	}
}