import java.lang.Math;
import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;

public class Test {
	public static void main(String[] args) {
		HashMap<Integer, Integer> nodesPerLayer = new HashMap<>();
		nodesPerLayer.put(0, 1);
		nodesPerLayer.put(1, 5);
		nodesPerLayer.put(2, 1);
		NeuralNetwork net = new NeuralNetwork(nodesPerLayer);
		testSine(net, 100, 10000);
		ArrayList<Double> test = new ArrayList<>();
		test.add(Math.PI / 6); // sin(pi/6) = 0.5
		System.out.println(net.feedforward(test));
		// ArrayList<Double> test2 = new ArrayList<>();
		// test2.add(Math.PI / 3);
		// for (int i = 0; i < 100; i++) {
		// 	net.backpropagate(test, 0.5, 0.25);
		// 	net.backpropagate(test2, 0.8660254037844386467637231707529361834714026269051903, 0.25);
		// }
		// System.out.println(net.feedforward(test));
		// System.out.println(net.feedforward(test2));
		// // ArrayList<Double> test3 = new ArrayList<>();
		// // for (int i = 0; i < 50; i++) {
		// // 	ArrayList<Double> temp = new ArrayList<>();
		// // 	temp.add(Math.random() * 100);
		// // 	System.out.println(net.feedforward(temp));
		// // }
	}

	public static void testSine(NeuralNetwork network, int size, int max_iterations) {
		double[][] inputs = new double[size][1];
		double[] expected = new double[size];
		for (int i = 0; i < size; i++) {
			inputs[i][0] = Math.random() * Math.PI;
			expected[i] = Math.sin(inputs[i][0]);
		}
		network.train(inputs, expected, max_iterations, 0.25);
	}

	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
}