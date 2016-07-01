public class Neuron {
	double output;

	public double sigmoid_function(double x) {
		return 1 / (1 + Math.exp(-x));
	}
}