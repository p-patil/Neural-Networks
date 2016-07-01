import java.util.HashMap;
import java.util.ArrayList;

public class TestNeuralNetwork {
	public static void main(String[] args) {
		HashMap<Integer, Integer> nodesByLayer = new HashMap<>();
		nodesByLayer.put(0, 3);
		nodesByLayer.put(1, 2);
		nodesByLayer.put(2, 3);
		NeuralNetwork network = new NeuralNetwork(3, nodesByLayer);
		double[] inputs = {1.0, 0.25, -0.5};
		double[] expected = {1.0, -1.0, 0.0};
		NeuralNetwork.TrainingSample sample = network.new TrainingSample(inputs, expected);
		System.out.println("OUTPUTS - Cycle 0");
		for (double d : network.run(inputs)) {
			System.out.println(d);
		}
		System.out.println();
		for (int i = 1; i < 20; i++) {
			// network.printWeights();
			network.train(sample);
			System.out.println("OUTPUTS - Cycle " + i);
			for (double d : network.run(inputs)) {
				System.out.println(d);
			}
			System.out.println();
		}
	}
}