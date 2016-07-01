import java.util.ArrayList;

public class BiasNeuron extends InputNeuron {
	public BiasNeuron() {
		super(-1);
	}

	@Override
	public double output_function(ArrayList<Double> inputVector) {
		return 1.0;
	}
}