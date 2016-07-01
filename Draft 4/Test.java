import java.util.ArrayList;
import java.util.HashMap;
import java.util.Arrays;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Test {
	public static void main(String[] args) {
		int[] nodesPerLayer = {1, 20, 1};
		NeuralNetwork net = new NeuralNetwork(nodesPerLayer);
		functionTest(net);
		// digitRecognitionTest();
	}

	public static void functionTest(NeuralNetwork net) {
		if (net.inputLayer.size() != 1 && net.outputLayer.size() != 1) {
			throw new IllegalArgumentException("The inputted neural network must accept one input and output one output");
		}
		System.out.println("Sine Test: f(x) = sin(x)");
		net.clear();
		net = sineTest(net, 10000, 100);
		ArrayList<Double> test1 = new ArrayList<>();
		ArrayList<Double> test2 = new ArrayList<>();
		ArrayList<Double> test3 = new ArrayList<>();
		ArrayList<Double> test4 = new ArrayList<>();
		test1.add(Math.PI / 6); // expect 0.5
		test2.add(Math.PI / 3); // expect 0.866025
		test3.add(Math.PI / 2); // expect 1
		test4.add(Math.PI / 4); // expect 0.707107
		System.out.println("sin(pi / 6): Got " + (2 * (net.feedforward(test1).get(0)) - 1) + ", Expected 0.5");
		System.out.println("sin(pi / 3): Got " + (2 * (net.feedforward(test2).get(0)) - 1) + ", Expected ~0.866025");
		System.out.println("sin(pi / 2): Got " + (2 * (net.feedforward(test3).get(0)) - 1) + ", Expected 1");
		System.out.println("sin(pi / 4): Got " + (2 * (net.feedforward(test4).get(0)) - 1) + ", Expected ~0.707107");
		System.out.println();
		System.out.println("Exponential Test: f(x) = e^x");
		net.clear();
		net = expTest(net, 10000, 100);
		test1.set(0, 1.0); // expect 2.7182818
		test2.set(0, 1.5); // expect 4.48169
		test3.set(0, 2.0); // expect 7.3890560
		test4.set(0, 2.5); // expect 12.1825
		System.out.println("e ^ 1: Got " + (((Math.exp(3) + 1) * net.feedforward(test1).get(0)) - 1) + ", Expected ~2.7182818");
		System.out.println("e ^ 1.5: Got " + (((Math.exp(3) + 1) * net.feedforward(test2).get(0)) - 1) + ", Expected ~4.48169");
		System.out.println("e ^ 2: Got " + (((Math.exp(3) + 1) * net.feedforward(test3).get(0)) - 1) + ", Expected ~7.3890560");
		System.out.println("e ^ 2.5: Got " + (((Math.exp(3) + 1) * net.feedforward(test4).get(0)) - 1) + ", Expected ~12.1825");
		System.out.println();
		System.out.println("Semi-Circle Test: f(x) = sqrt(1 - x^2)");
		net.clear();
		net = semicircleTest(net, 10000, 100);
		test1.set(0, 0.0); // expect 1
		test2.set(0, Math.sqrt(3) / 2); // expect 0.5
		test3.set(0, 0.5); // expect 0.866025
		test4.set(0, 0.75); // expect 0.661438
		System.out.println("sqrt(1 - 0^2): Got " + ((2 * net.feedforward(test1).get(0)) - 1) + ", Expected 1");
		System.out.println("sqrt(1 - (sqrt(3) / 2)^2): Got " + ((2 * net.feedforward(test2).get(0)) - 1) + ", Expected 0.5");
		System.out.println("sqrt(1 - (1/2)^2): Got " + ((2 * net.feedforward(test3).get(0)) - 1) + ", Expected 0.866025");
		System.out.println("sqrt(1 - 0^2): Got " + ((2 * net.feedforward(test4).get(0)) - 1) + ", Expected 0.661438");
		System.out.println();
		System.out.println("Parabola Test: f(x) = 5x^2 - 4x + 1");
		net.clear();
		net = parabolaTest(net, 10000, 100);
		test1.set(0, 0.0); // expect 1
		test2.set(0, 2.0); // expect 13
		test3.set(0, 0.5); // expect 0.25
		test4.set(0, -3.0); // expect 58
		System.out.println("5 * (0)^2 - 4 * (0) + 1: Got " + ((102.0 * net.feedforward(test1).get(0)) - 5) + ", Expected 1");
		System.out.println("5 * (2)^2 - 4 * (2) + 1: Got " + ((102.0 * net.feedforward(test2).get(0)) - 5) + ", Expected 13");
		System.out.println("5 * (1/2)^2 - 4 * (1/2) + 1: Got " + ((102.0 * net.feedforward(test3).get(0)) - 5) + ", Expected 0.25");
		System.out.println("5 * (-3)^2 - 4 * (-3) + 1: Got " + ((102.0 * net.feedforward(test4).get(0)) - 5) + ", Expected 58");
		System.out.println();
		System.out.println("Parabola Test: f(x) = nearest integer to x");
		net.clear();
		net = roundTest(net, 10000, 100);
		test1.set(0, 5.67); // expect 5
		test2.set(0, 9.21); // expect 9
		test3.set(0, 0.441); // expect 0
		test4.set(0, 0.576); // expect 1
		System.out.println("Nearest integer to 5.67: Got " + (10.0 * net.feedforward(test1).get(0)) + ", Expected 5");
		System.out.println("Nearest integer to 9.21: Got " + (10.0 * net.feedforward(test2).get(0)) + ", Expected 9");
		System.out.println("Nearest integer to 0.441: Got " + (10.0 * net.feedforward(test3).get(0)) + ", Expected 0");
		System.out.println("Nearest integer to 0.576: Got " + (10.0 * net.feedforward(test4).get(0)) + ", Expected 1");
	}

	public static NeuralNetwork digitRecognitionTest() {
		try {
			String mnist_path = "C:/Users/vip/Documents/Files/Other/machine learning/neural networks/Testing/MNIST Database of Handwritten Digits/mnist_train.txt";
			// int num_images = 60000;
			int num_images = 1000;
			int image_width = 28;
			int image_height = 28;
			int[] nodesPerLayer = {image_height * image_width, 15, 10};
			NeuralNetwork net = new NeuralNetwork(nodesPerLayer);
			double[][] inputs = new double[num_images][image_height * image_width];
			double[][] expected = new double[num_images][10];
			BufferedReader reader = new BufferedReader(new FileReader(new File(mnist_path)));
			String line = reader.readLine();
			int i = 0;
			while ((line = reader.readLine()) != null) {
				if (i >= num_images) {
					break;
				}
				String[] image = line.split(",");
				int label = Integer.parseInt(image[0]);
				expected[i][label] = 1;
				for (int j = 1; j < image.length; j++) {
					inputs[i][j - 1] = Double.parseDouble(image[j]);
				}
				i++;
			}
			System.out.print("Training... ");
			long start = System.nanoTime();
			net.train(inputs, expected, 0.1, 1000);
			long finish = System.nanoTime();
			System.out.println("Done. Training took " + getTimeElapsed(start, finish));
			String test_path = "C:/Users/vip/Documents/Files/Other/machine learning/neural networks/Testing/MNIST Database of Handwritten Digits/mnist_test.txt";
			reader = new BufferedReader(new FileReader(new File(test_path)));
			line = reader.readLine();
			double correct = 0.0;
			double total = 0.0;
			while ((line = reader.readLine()) != null) {
				String[] image = line.split(",");
				int label = Integer.parseInt(image[0]);
				ArrayList<Double> test_input = new ArrayList<>();
				for (int j = 1; j < image.length; j++) {
					test_input.add(Double.parseDouble(image[j]));
				}
				ArrayList<Double> actual = net.feedforward(test_input);
				int max_index = 0;
				for (int j = 0; j < actual.size(); j++) {
					if (actual.get(j) > actual.get(max_index)) {
						max_index = j;
					}
				}
				if (max_index == label) {
					correct++;
				}
				total++;
			}
			System.out.println("Accuracy: " + ((correct / total) * 100) + "%");
			return net;
		} catch (IOException e) {
			System.out.println("File not found.");
			return null;
		}
	}

	// HELPER METHODS BELOW THIS LINE.

	private static String getTimeElapsed(long start, long finish) {
		double seconds = (finish - start) * 0.000000001;
		String time = seconds + " seconds";
		if (seconds > 60) {
			int minutes = (int) (seconds / 60);
			seconds %= 60;
			time = minutes + " minutes " + seconds + " seconds"; 
			if (minutes == 1) {
				time = minutes + " minute " + seconds + " seconds";
			}
			if (minutes > 60) {
				int hours = (int) (minutes / 60);
				minutes %= 60;
				time = hours + " hours " + minutes + " minutes " + seconds + " seconds";
				if (hours == 1) {
					time = hours + " hour " + minutes + " minutes " + seconds + " seconds";					
				}
				if (hours > 24) {
					int days = (int) (hours / 24);
					hours %= 24;
					time = days + " days " + hours + " hours " + minutes + " minutes " + seconds + " seconds ";
					if (days == 1) {
						time = days + " day " + hours + " hours " + minutes + " minutes " + seconds + " seconds ";
					}
				}
			}
		}
		return time;
	}

	private static NeuralNetwork sineTest(NeuralNetwork network, int max_iterations, int trainingSetSize) {
		double[][] inputs = new double[trainingSetSize][1];
		double[][] expected = new double[trainingSetSize][1];
		int i = 0;
		for (double x : sin_domain(trainingSetSize)) {
			inputs[i][0] = x;
			expected[i][0] = modified_sin(x);
			i++;
		}
		System.out.print("Training... ");
		long start = System.nanoTime();
		network.train(inputs, expected, 0.7, max_iterations);
		// network.trainWithMomentum(inputs, expected, max_iterations);
		// network.trainWithAdaptingLearning_BoldDriver(inputs, expected, max_iterations);
		// network.trainWithAdaptingLocalLearning(inputs, expected, max_iterations);
		long finish = System.nanoTime();
		System.out.println("Done. Training took " + getTimeElapsed(start, finish));;
		ArrayList<Double> errors = new ArrayList<>();
		for (double x : sin_domain(trainingSetSize)) {
			ArrayList<Double> input_x = new ArrayList<>();
			input_x.add(x);
			errors.add(Math.abs(modified_sin(x) - network.feedforward(input_x).get(0)));
		}
		double avg_error = 0.0;
		for (double err : errors) {
			avg_error += err;
		}
		avg_error /= errors.size();
		System.out.println("Avg error: " + avg_error);
		return network;
	}

	private static double modified_sin(double x) {
		return 0.5 * (1.0 + Math.sin(x));
	}

	private static ArrayList<Double> sin_domain(int n) {
		ArrayList<Double> ret = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			// ret.add(Math.random() * Math.PI * 4);
			ret.add(Math.random() * Math.PI);
		}
		return ret;
	}

	private static NeuralNetwork expTest(NeuralNetwork network, int max_iterations, int trainingSetSize) {
		double[][] inputs = new double[trainingSetSize][1];
		double[][] expected = new double[trainingSetSize][1];
		int i = 0;
		for (double x : exp_domain(trainingSetSize)) {
			inputs[i][0] = x;
			expected[i][0] = modified_exp(x);
			i++;
		}
		System.out.print("Training... ");
		long start = System.nanoTime();
		network.train(inputs, expected, 0.7, max_iterations);
		long finish = System.nanoTime();
		System.out.println("Done. Training took " + getTimeElapsed(start, finish));;
		ArrayList<Double> errors = new ArrayList<>();
		for (double x : exp_domain(trainingSetSize)) {
			ArrayList<Double> input_x = new ArrayList<>();
			input_x.add(x);
			errors.add(Math.abs(modified_exp(x) - network.feedforward(input_x).get(0)));
		}
		double avg_error = 0.0;
		for (double err : errors) {
			avg_error += err;
		}
		avg_error /= errors.size();
		System.out.println("Avg error: " + avg_error);
		return network;
	}

	private static double modified_exp(double x) {
		return (1.0 / (Math.exp(3) + 1)) * (Math.exp(x) + 1);
	}

	private static ArrayList<Double> exp_domain(int n) {
		ArrayList<Double> ret = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			ret.add(Math.random() * 3);
		}
		return ret;
	}

	private static NeuralNetwork semicircleTest(NeuralNetwork network, int max_iterations, int trainingSetSize) {
		double[][] inputs = new double[trainingSetSize][1];
		double[][] expected = new double[trainingSetSize][1];
		int i = 0;
		for (double x : semicircle_domain(trainingSetSize)) {
			inputs[i][0] = x;
			expected[i][0] = modified_semicircle(x);
			i++;
		}
		System.out.print("Training... ");
		long start = System.nanoTime();
		network.train(inputs, expected, 0.7, max_iterations);
		long finish = System.nanoTime();
		System.out.println("Done. Training took " + getTimeElapsed(start, finish));;
		ArrayList<Double> errors = new ArrayList<>();
		for (double x : semicircle_domain(trainingSetSize)) {
			ArrayList<Double> input_x = new ArrayList<>();
			input_x.add(x);
			errors.add(Math.abs(modified_semicircle(x) - network.feedforward(input_x).get(0)));
		}
		double avg_error = 0.0;
		for (double err : errors) {
			avg_error += err;
		}
		avg_error /= errors.size();
		System.out.println("Avg error: " + avg_error);
		return network;
	}

	private static double modified_semicircle(double x) {
		double semicircle_value = Math.sqrt(1 - Math.pow(x, 2));
		return 0.5 * (semicircle_value + 1);
	}

	private static ArrayList<Double> semicircle_domain(int n) {
		ArrayList<Double> ret = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			ret.add(2 * (Math.random() - 0.5));
		}
		return ret;
	}

	private static NeuralNetwork parabolaTest(NeuralNetwork network, int max_iterations, int trainingSetSize) {
		double[][] inputs = new double[trainingSetSize][1];
		double[][] expected = new double[trainingSetSize][1];
		int i = 0;
		for (double x : parabola_domain(trainingSetSize)) {
			inputs[i][0] = x;
			expected[i][0] = modified_parabola(x);
			i++;
		}
		System.out.print("Training... ");
		long start = System.nanoTime();
		network.train(inputs, expected, 0.7, 1000);
		long finish = System.nanoTime();
		System.out.println("Done. Training took " + getTimeElapsed(start, finish));;
		ArrayList<Double> errors = new ArrayList<>();
		for (double x : parabola_domain(trainingSetSize)) {
			ArrayList<Double> input_x = new ArrayList<>();
			input_x.add(x);
			errors.add(Math.abs(modified_parabola(x) - network.feedforward(input_x).get(0)));
		}
		double avg_error = 0.0;
		for (double err : errors) {
			avg_error += err;
		}
		avg_error /= errors.size();
		System.out.println("Avg error: " + avg_error);
		return network;
	}

	private static double modified_parabola(double x) {
		return (1.0 / 102.0) * ((5 * Math.pow(x, 2) - 4 * x + 1) + 5);
	}

	private static ArrayList<Double> parabola_domain(int n) {
		ArrayList<Double> ret = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			ret.add(8 * (Math.random() - 0.5));
		}
		return ret;
	}

	private static NeuralNetwork roundTest(NeuralNetwork network, int max_iterations, int trainingSetSize) {
		double[][] inputs = new double[trainingSetSize][1];
		double[][] expected = new double[trainingSetSize][1];
		int i = 0;
		for (double x : round_domain(trainingSetSize)) {
			inputs[i][0] = x;
			expected[i][0] = modified_round(x);
			i++;
		}
		System.out.print("Training... ");
		long start = System.nanoTime();
		network.train(inputs, expected, 0.7, max_iterations);
		long finish = System.nanoTime();
		System.out.println("Done. Training took " + getTimeElapsed(start, finish));;
		ArrayList<Double> errors = new ArrayList<>();
		for (double x : round_domain(trainingSetSize)) {
			ArrayList<Double> input_x = new ArrayList<>();
			input_x.add(x);
			errors.add(Math.abs(modified_round(x) - network.feedforward(input_x).get(0)));
		}
		double avg_error = 0.0;
		for (double err : errors) {
			avg_error += err;
		}
		avg_error /= errors.size();
		System.out.println("Avg error: " + avg_error);
		return network;
	}

	private static double modified_round(double x) {
		return (1 / 10.0) * ((int) (x + 0.5));
	}

	private static ArrayList<Double> round_domain(int n) {
		ArrayList<Double> ret = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			ret.add(10 * (Math.random()));
		}
		return ret;
	}
}