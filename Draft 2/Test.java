import java.util.HashMap;
import java.util.HashSet;
import java.lang.Math;

public class Test {
	public static void main(String[] args) {
		HashMap<Integer, Integer> nodes = new HashMap<>();
		nodes.put(0, 1);
		nodes.put(1, 3);
		nodes.put(2, 1);
		NeuralNetwork net = new NeuralNetwork(3, nodes);
		double[] inputs = {Math.PI / 6};
		// for (double ans : net.feedforward(inputs)) {
		// 	System.out.println(ans);
		// }
		// net.print();
		// testSine(net, 100, 100);
		// net.print();
		// for (double ans : net.feedforward(inputs)) {
		// 	System.out.println(ans); // expect 2.5
		// }
		net.feedforward(inputs);
		net.print();
		testSine(net, 100, 10000);
	}

	public static double sigma(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static void testSine(NeuralNetwork network, int size, int max_iterations) {
		double[][] inputs = new double[size][1];
		double[][] expected = new double[size][1];
		HashSet<Double> xs = new HashSet<Double>();
		for (int i = 0; i < size; i++) {
			double x = 0.0;
			do {
				x = Math.random() * Math.PI;
			} while (xs.contains(x));
			xs.add(x);
			double y = Math.sin(x);
			inputs[i][0] = x;
			expected[i][0] = y;
		}
		network.train(inputs, expected, max_iterations);
	}
}