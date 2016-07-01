public class HiddenNeuron extends Neuron {
	double[] input_weight_product;

	public double activation_function() {
		double ret = 0.0;
		for (double i : input_weight_product) {
			ret += i;
		}
		return ret;
	}

	public double output_function() {
		this.output = sigmoid_function(activation_function());
		return this.output;
	}

	public HiddenNeuron() {

	}
}