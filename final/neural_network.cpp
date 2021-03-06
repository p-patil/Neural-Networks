#include <random>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <cstdarg>
#include "neural_network.h"

using namespace std;

/* Begin Neuron class. */

/* Sigmoid activation function: f(x) = 1 / (1 + e^(- lambda * x)) */

Neuron::Neuron() {}

/* Copy constructor. */
Neuron::Neuron(const Neuron &n) { *this = n; }

/* Basic constructor; randomly initializes weights. */
Neuron::Neuron(Neuron **synapses, int length) {
    this->num_synapses = length;
    this->synapses = new Neuron * [length];
    this->weights = new double[length];
    
    for (int i = 0; i < length; ++i) {
        this->synapses[i] = synapses[i];
        this->weights[i] = 2 * (((double) rand()) / RAND_MAX) - 1;
    }
}

Neuron::Neuron(Neuron **synapses, double *weights, int length) {
    this->num_synapses = length;
    this->synapses = new Neuron * [length];
    this->weights = new double[length];
    
    for (int i = 0; i < length; ++i) {
        this->synapses[i] = synapses[i];
        this->weights[i] = weights[i];
    }
}

Neuron ** Neuron::get_synapses(void) const { return this->synapses; }

double * Neuron::get_weights(void) const { return this->weights; }

double Neuron::get_weight(int i) const { return this->weights[i]; }

void Neuron::set_weight(int i, double val) { this->weights[i] = val; }

int Neuron::get_num_synapses(void) const { return this->num_synapses; }

bool Neuron::is_output_neuron(void) const { return false; }

/* Assignment operator. */
Neuron & Neuron::operator =(const Neuron & n) {
    this->num_synapses = n.get_num_synapses();

    delete[] this->synapses;
    delete[] this->weights;

    this->synapses = new Neuron *[this->num_synapses];
    this->weights = new double[this->num_synapses];

    Neuron **synapses = n.get_synapses();
    double *weights = n.get_weights();
    for (int i = 0; i < this->num_synapses; ++i) {
        this->synapses[i] = synapses[i];
        this->weights[i] = weights[i];
    }

    return *this;
}

bool Neuron::operator ==(const Neuron &n) {
    if (this->num_synapses != n.get_num_synapses()) {
        return false;
    }

    double *weights = n.get_weights();
    for (int i = 0; i < this->num_synapses; ++i) {
        if (this->weights[i] != weights[i]) {
            return false;
        }
    }

    return true;
}

/* Destructor. */
Neuron::~Neuron(void) {
    delete[] this->synapses;
    delete[] this->weights;
}

/* End Neuron class. */

/* Begin BiasNeuron class. */

BiasNeuron::BiasNeuron(Neuron **synapses, int length) : Neuron(synapses, length) {}

BiasNeuron::BiasNeuron(Neuron **synapses, double *weights, int length) : Neuron(synapses, weights, length) {}

/* End BiasNeuron class. */

/* Begin OutputNeuron class. */

OutputNeuron::OutputNeuron() {}

bool OutputNeuron::is_output_neuron(void) const { return true; }

bool OutputNeuron::operator ==(const Neuron &n) { return n.is_output_neuron(); }

/* End OutputNeuron class. */

/* Begin NeuralNetwork class. Note: The implementation below makes use of the sigmoid activation function,
 * whose range is [0, 1], and thus the resulting net is unable to output negative values. Thus, it's
 * advised to normalize input vectors to [0, 1]^n before training. */

// Constructors

/* Initializes a neural network, with random weights, whose i-th layer has layer_counts[i] neurons. */
NeuralNetwork::NeuralNetwork(int *layer_counts, int length) {
    /* Initialize. */
    this->num_layers = length;
    this->layer_counts = new int[length];
    this->layers = new Neuron **[length];
    this->bias_neurons = new BiasNeuron *[length - 1];

    /* Copy layer counts over. */
    for (int i = 0; i < length; ++i) {
        this->layer_counts[i] = layer_counts[i];
    }

    /* Construct layers in backwards order, connecting synapses along the way. Start with output layer. */
    this->layers[length - 1] = new Neuron *[layer_counts[length - 1]];
    for (int i = 0; i < layer_counts[length - 1]; ++i) {
        this->layers[length - 1][i] = new OutputNeuron();
    }

    for (int i = length - 2; i >= 0; --i) {
        this->layers[i] = new Neuron *[layer_counts[i]];
        for (int j = 0; j < layer_counts[i]; ++j) {
            this->layers[i][j] = new Neuron(this->layers[i + 1], layer_counts[i + 1]);
        }

        this->bias_neurons[i] = new BiasNeuron(this->layers[i + 1], layer_counts[i + 1]);
    }
}

/* Variadic constructor that accepts an arbitrary number (at least 1) of arguments, the first of which specifies the number of layers
 * with the rest specifying the number of neurons in the respective layers. */
NeuralNetwork::NeuralNetwork(int num_layers, ...) {
    va_list layer_counts;
    va_start(layer_counts, num_layers);

    /* Initialize. */
    this->num_layers = num_layers;
    this->layer_counts = new int[num_layers];
    this->layers = new Neuron **[num_layers];
    this->bias_neurons = new BiasNeuron *[num_layers - 1];
    
    /* Copy layer counts over. */
    for (int i = 0; i < num_layers; ++i) {
        this->layer_counts[i] = va_arg(layer_counts, int); 
    }

    va_end(layer_counts);

    /* Construct layers in backwards order, connecting synapses along the way. Start with output layer. */
    this->layers[num_layers - 1] = new Neuron *[this->layer_counts[num_layers - 1]];
    for (int i = 0; i < this->layer_counts[num_layers - 1]; ++i) {
        this->layers[num_layers - 1][i] = new OutputNeuron();
    }

    for (int i = num_layers - 2; i >= 0; --i) {
        this->layers[i] = new Neuron *[this->layer_counts[i]];
        for (int j = 0; j < this->layer_counts[i]; ++j) {
                this->layers[i][j] = new Neuron(this->layers[i + 1], this->layer_counts[i + 1]);
        }

        this->bias_neurons[i] = new BiasNeuron(this->layers[i + 1], this->layer_counts[i + 1]);
    }
}

/* Initializes a neural network with the given weights whose i-th layer has layerCounts[i] neurons. The weights matrix
 * should have (length - 1) rows. */
NeuralNetwork::NeuralNetwork(int length, int *layer_counts, double ***weights, double **bias_weights) { 
    /* Initialize. */
    this->num_layers = length;
    this->layer_counts = new int[length];
    this->layers = new Neuron **[length];
    this->bias_neurons = new BiasNeuron *[length - 1];

    /* Copy layer counts over. */
    for (int i = 0; i < length; ++i) {
        this->layer_counts[i] = layer_counts[i];
    }

    /* Construct layers in backwards order, connecting synapses along the way. Start with output layer. */
    this->layers[length - 1] = new Neuron *[layer_counts[length - 1]];
    for (int i = 0; i < layer_counts[length - 1]; ++i) {
        this->layers[length - 1][i] = new OutputNeuron();
    }

    for (int i = length - 2; i >= 0; --i) {
        this->layers[i] = new Neuron *[layer_counts[i]];
        for (int j = 0; j < layer_counts[i]; ++j) {
            this->layers[i][j] = new Neuron(this->layers[i + 1], weights[i][j], layer_counts[i + 1]);
        }

        this->bias_neurons[i] = new BiasNeuron(this->layers[i + 1], bias_weights[i], layer_counts[i + 1]);
    }
}

// Utility functions

void NeuralNetwork::initialize_weight_array(ARRAY_3D &arr, const NeuralNetwork &net) {
    arr = ARRAY_3D(net.get_num_layers() - 1);
    for (int i = 0; i < arr.size(); ++i) {
                arr[i] = ARRAY_2D(net.get_layer_count(i));
        for (int j = 0; j < arr[i].size(); ++j) {
            arr[i][j] = ARRAY(net.get_layer_count(i + 1));
                        for (int k = 0; k < arr[i][j].size(); ++k) {
                            arr[i][j][k] = 0.0;
                        }
        }
    }
}

void NeuralNetwork::initialize_bias_array(ARRAY_2D &arr, const NeuralNetwork &net ) {
    arr = ARRAY_2D(net.get_num_layers() - 1);
    for (int i = 0; i < arr.size(); ++i) {
        arr[i] = ARRAY(net.get_layer_count(i + 1));
    for (int j = 0; j < arr[i].size(); ++j) {
            arr[i][j] = 0.0; 
    }
    }
}

void NeuralNetwork::add_arrays(ARRAY_2D &arr, const ARRAY_2D &summand) {
    for (int i = 0; i < arr.size(); ++i) {
        for (int j = 0; j < arr[i].size(); ++j) {
            arr[i][j] += summand[i][j];
        }
    }
}

void NeuralNetwork::add_arrays(ARRAY_3D &arr, const ARRAY_3D &summand) {
    for (int i = 0; i < arr.size(); ++i) {
        for (int j = 0; j < arr[i].size(); ++j) {
            for (int k = 0; k < arr[i][j].size(); ++k) {
                arr[i][j][k] += summand[i][j][k];
            }
        }
    }
}

double NeuralNetwork::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

double NeuralNetwork::sigmoid_derivative_input(double x) {
    double y = sigmoid(x);
    return y * (1 - y);
}

double NeuralNetwork::sigmoid_derivative(double y) { return y * (1 - y); }

double NeuralNetwork::quadratic_error(double output, double expected) {
    double diff = output - expected;
    return 0.5 * diff * diff; 
}

double NeuralNetwork::quadratic_error_derivative(double output, double expected) { return output - expected; }

double NeuralNetwork::step_decay(double learning_rate, double reduce_factor) { return learning_rate - reduce_factor; }

double NeuralNetwork::exponential_decay(double alpha_0, double k, double t) { return alpha_0 * exp(-k * t); }

double NeuralNetwork::one_over_t_decay(double alpha_0, double k, double t) { return alpha_0 / (1 + k * t); }

// Getters

int NeuralNetwork::get_num_layers(void) const { return this->num_layers; }

int NeuralNetwork::get_layer_count(int index) const { return this->layer_counts[index]; }

/* Returned array was allocated to heap with new[]. */
int * NeuralNetwork::get_layer_counts(void) const {
    int *ret = new int[this->get_num_layers()];
    for (int i = 0; i < this->get_num_layers(); ++i) {
        ret[i] = this->get_layer_count(i);
    }

    return ret;
}

Neuron * NeuralNetwork::get_neuron(int i, int j) const { return this->layers[i][j]; }

BiasNeuron * NeuralNetwork::get_bias_neuron(int i) const { return this->bias_neurons[i]; }

double NeuralNetwork::get_weight(int i, int j, int k) const { return this->get_neuron(i, j)->get_weight(k); }

void NeuralNetwork::set_weight(int i, int j, int k, double val) { this->get_neuron(i, j)->set_weight(k, val); }

double NeuralNetwork::get_bias_weight(int i, int j) const { return this->get_bias_neuron(i)->get_weight(j); }

void NeuralNetwork::set_bias_weight(int i, int j, double val) { this->get_bias_neuron(i)->set_weight(j, val); }
// Algorithm code

/* Given an actual output and expected output, returns all the deltas (one per non-output neuron) according to the delta rule. 
 * The returned array's indices are shifted backwards by 1, so that ret[i] represents layer (i + 1) of this neural network. */
ARRAY_2D NeuralNetwork::backpropagate(const ARRAY &output, const ARRAY &expected, const ARRAY_2D &neuron_outputs) const {
    int length = this->get_num_layers();
    ARRAY_2D deltas (length);

    /* Initialize data for output layer. */
    deltas[length - 1] = ARRAY(this->get_layer_count(this->get_num_layers() - 1));
    for (int i = 0; i < deltas[length - 1].size(); ++i) {
        deltas[length - 1][i] = this->activation_function_derivative(output[i]) * this->error_function_derivative(output[i], expected[i]);
    }

    /* Backwards propagate to recursively compute each delta:
     * delta[i][j] = (1 - o_{i, j}) * o_{i, j} * (sum over neurons n in next layer (delta_{i + 1, n} * w))
     * where w = weight between neuron j in layer i and neuron n in layer i + 1. */
    for (int i = length - 2; i >= 1; --i) {
        deltas[i] = ARRAY(this->get_layer_count(i));

        for (int j = 0; j < deltas[i].size(); ++j) {
            deltas[i][j] = 0.0;

            for (int k = 0; k < this->get_layer_count(i + 1); ++k) {
                deltas[i][j] += deltas[i + 1][k] * this->get_weight(i, j, k);
            }

            deltas[i][j] *= this->activation_function_derivative(neuron_outputs[i][j]);
        }
    }

    /* Input neurons have no associated delta value, so delta[0] is never set. */

    return deltas;
}

ARRAY_3D NeuralNetwork::compute_derivatives(const ARRAY_2D &outputs, const ARRAY_2D &deltas) const {
    ARRAY_3D derivatives (this->get_num_layers() - 1);

    for (int i = 0; i < this->get_num_layers() - 1; ++i) {
        derivatives[i] = ARRAY_2D(this->get_layer_count(i));

        for (int j = 0; j < this->get_layer_count(i); ++j) {
            derivatives[i][j] = ARRAY(this->get_layer_count(i + 1));

            for (int k = 0; k < this->get_layer_count(i + 1); ++k) {
                derivatives[i][j][k] = deltas[i + 1][k] * outputs[i][j];
            }
        }
    }

    return derivatives;
}

ARRAY_2D NeuralNetwork::compute_bias_derivatives(const ARRAY_2D &deltas) const {
    ARRAY_2D bias_derivatives (this->get_num_layers() - 1);

    for (int i = 0; i < this->get_num_layers() - 1; ++i) {
        bias_derivatives[i] = ARRAY(this->get_layer_count(i + 1));

        for (int j = 0; j < this->get_layer_count(i + 1); ++j) {
            bias_derivatives[i][j] = deltas[i + 1][j]; // output of bias neurons is always 1
        }
    }

    return bias_derivatives;
}

/* Updates weights by shifting in the direction opposite the average error function gradient. */
ARRAY_3D  NeuralNetwork::update_weights(const ARRAY_3D &gradient, const ARRAY_3D &prev_deltas, double learning_rate, double momentum) {
        ARRAY_3D deltas (this->get_num_layers() - 1);
    double weight, delta_weight;
    for (int i = 0; i < this->get_num_layers() - 1; ++i) {
                deltas[i] = ARRAY_2D(this->get_layer_count(i));

        for (int j = 0;  j < this->get_layer_count(i); ++j) {
                        deltas[i][j] = ARRAY(this->get_layer_count(i + 1));

            for (int k = 0; k < this->get_layer_count(i + 1); ++k) {
                weight = this->get_weight(i, j, k);
                delta_weight = - learning_rate * gradient[i][j][k] + prev_deltas[i][j][k] * momentum;

                deltas[i][j][k] = delta_weight;
                this->set_weight(i, j, k, weight + delta_weight);
            }
        }
    }

        return deltas;
}

ARRAY_2D NeuralNetwork::update_bias_weights(const ARRAY_2D &bias_gradient, const ARRAY_2D &prev_deltas, double learning_rate, double momentum) {
    ARRAY_2D deltas (this->get_num_layers() - 1);
    double weight, delta_weight;
    for (int i = 0; i < this->get_num_layers() - 1; ++i) {
                deltas[i] = ARRAY(this->get_layer_count(i + 1));

        for (int j = 0;  j < this->get_layer_count(i + 1); ++j) {
            weight = this->get_bias_weight(i, j);
            delta_weight = - learning_rate * bias_gradient[i][j] + prev_deltas[i][j] * momentum;

            deltas[i][j] = delta_weight;
            this->set_bias_weight(i, j, weight + delta_weight);
        }
    }

        return deltas;
}

/* Given a training batch fills the given arrays with the sums of all derivatives of the error function with respect to each weight. */
pair<ARRAY_3D, ARRAY_2D> NeuralNetwork::gradient_descent(pair<ARRAY, ARRAY> *batch, int batch_size)  const {
    ARRAY output, expected;
    ARRAY_2D avg_bias_derivatives = ARRAY_2D(this->get_num_layers() - 1);
    ARRAY_3D avg_derivatives = ARRAY_3D(this->get_num_layers() - 1);

    initialize_weight_array(avg_derivatives, *this);
    initialize_bias_array(avg_bias_derivatives, *this);

    /* Sum all the error function gradients for (input, output) pairs in this batch. */
    ARRAY_2D neuron_outputs, deltas, bias_derivatives;
    ARRAY_3D derivatives;
    for (int i = 0; i < batch_size; ++i) {
        neuron_outputs = this->feedforward_and_get_outputs(get<0>(batch[i]));
        output = neuron_outputs[this->get_num_layers() - 1];
        expected = get<1>(batch[i]);

        /* Backpropagate to compute deltas. */
        deltas = this->backpropagate(output, expected, neuron_outputs);

        /* Compute the gradients. */
        derivatives = this->compute_derivatives(neuron_outputs, deltas);
        bias_derivatives = this->compute_bias_derivatives(deltas); // No need to pass bias outputs since they're always 1

        /* Add to running total. */
        NeuralNetwork::add_arrays(avg_derivatives, derivatives);
        NeuralNetwork::add_arrays(avg_bias_derivatives, bias_derivatives);
    }


    /* Normalize the derivative sums to get the average. */
    for (int i = 0; i < avg_derivatives.size(); ++i) {
        for (int j = 0; j < avg_derivatives[i].size(); ++j) {
            for (int k = 0; k < avg_derivatives[i][j].size(); ++k) {
                avg_derivatives[i][j][k] /= batch_size;
            }
        }
    }

    for (int i = 0; i < avg_bias_derivatives.size(); ++i) {
        for (int j = 0; j < avg_bias_derivatives[i].size(); ++j) {
            avg_bias_derivatives[i][j] /= batch_size;
        }
    }

    return make_pair(avg_derivatives, avg_bias_derivatives);
}

// Functionality

ARRAY NeuralNetwork::feedforward(const ARRAY &inputs) const {
    ARRAY_2D outputs = this->feedforward_and_get_outputs(inputs);
    return outputs[outputs.size() - 1];
}

/* Feeds forward the given input vector as normal but returns an array storing the output of each neuron in the network. */
ARRAY_2D NeuralNetwork::feedforward_and_get_outputs(const ARRAY &inputs) const {
    ARRAY_2D outputs (this->get_num_layers());

    /* Initialize by copying the inputs over, since they are the data corresponding to the first layer. */
    outputs[0] = ARRAY(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        outputs[0][i] = inputs[i]; // Output of the input layer is simply the inputs
    }

    /* Feed the data in, at each layer computing the inputs to the next layer. */
    for (int i = 1; i < this->get_num_layers(); ++i) {
        /* For each neuron in the current layer, increment the input vector for neurons in the next layer with the output 
         * of the current neuron. */
        outputs[i] = ARRAY(this->get_layer_count(i));

        /* outputs[i][j] = f(W * outputs[i - 1]) - b) where f = activation function, W = weight matrix between previous layer and current 
         *                 layer, b = bias from previous layer. */
        for (int j = 0; j < this->get_layer_count(i); ++j) {
            outputs[i][j] = 1 * this->get_bias_weight(i - 1, j); // Output of bias neurons is always 1

            for (int k = 0; k < this->get_layer_count(i - 1); ++k) {
                outputs[i][j] += this->get_weight(i - 1, k, j) * outputs[i - 1][k];
            }

            outputs[i][j] = this->activation_function(outputs[i][j]);
        }
    }

    return outputs;
}

/* Trains this neural network on the given data. */
void NeuralNetwork::train(vector<pair<ARRAY, ARRAY>> samples, int num_epochs /* = 1 */, int batch_size /* = 1 */, double learning_rate /* = 0.7 */, double momentum /* = 0.1 */) {
    pair<ARRAY_3D, ARRAY_2D> p;
    ARRAY_3D avg_derivative, prev_weight_deltas;
    ARRAY_2D avg_bias_derivative, prev_bias_deltas;

    cout << endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        cout << "\tEpoch: " << epoch << endl;

        random_shuffle(samples.begin(), samples.end()); // Shuffle so batches are random
        initialize_weight_array(prev_weight_deltas, *this);
        initialize_bias_array(prev_bias_deltas, *this);

        for (int i = 0; i < samples.size(); i += batch_size) {
            /* Perform gradient descent on the batch to get the sum of derivatives. */
            p = this->gradient_descent(&samples[0] + i, batch_size);
            avg_derivative = get<0>(p);
            avg_bias_derivative = get<1>(p);

            /* Update weights using the gradient. */
            prev_weight_deltas = this->update_weights(avg_derivative, prev_weight_deltas, learning_rate, momentum);
            prev_bias_deltas = this->update_bias_weights(avg_bias_derivative, prev_bias_deltas, learning_rate, momentum);

            /* If using an adaptive learning rate and momentum, compute the next iteration. By default the below two
             * functions are simply identity functions. */
            learning_rate = this->adapt_learning_rate(learning_rate, epoch);
            momentum = this->adapt_momentum(momentum, epoch);
        }

        /* Train on the final mini-batch at the end of the training set. */
        if (samples.size() % batch_size != 0) {
            int mini_batch_size = (samples.size() / batch_size) * batch_size;
            p = this->gradient_descent(&samples[0] + (samples.size() - mini_batch_size), mini_batch_size);
            avg_derivative = get<0>(p);
            avg_bias_derivative = get<1>(p);

            /* Update weights using gradient. */
            this->update_weights(avg_derivative, prev_weight_deltas, learning_rate, momentum);
            this->update_bias_weights(avg_bias_derivative, prev_bias_deltas, learning_rate, momentum);
        }
    }
}

/* Sigmoid activation function. */
double NeuralNetwork::activation_function(double x) const { return sigmoid(x); }

/* Derivative of the activation function given an output. */
double NeuralNetwork::activation_function_derivative(double y) const { return sigmoid_derivative(y); }

/* Error function used in training: error(x, y) = 1/2 * (mag(x - y)^2) for vectors x, y. */
double NeuralNetwork::error_function(const double output, double expected) const { return quadratic_error(output, expected); }

double NeuralNetwork::error_function_derivative(double output, double expected) const { return quadratic_error_derivative(output, expected); }

double NeuralNetwork::adapt_learning_rate(double learning_rate, int epoch) const { return one_over_t_decay(learning_rate, 0.5, epoch); }

double NeuralNetwork::adapt_momentum(double momentum, int epoch) const { return momentum; }

/* Destructor. */
NeuralNetwork::~NeuralNetwork(void) {
    for (int i = 0; i < this->get_num_layers(); ++i) {
        for (int j = 0; j < this->get_layer_count(i); ++j) {
            delete this->layers[i][j];
        }

        delete[] this->layers[i];
    }

    for (int i = 0; i < this->get_num_layers() - 1; ++i) {
        delete this->bias_neurons[i];
    }

    delete[] this->bias_neurons;
    delete[] this->layer_counts;
}

/* End NeuralNetwork class. */
