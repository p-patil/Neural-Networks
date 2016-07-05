#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include "neural_network.h"

#define LENGTH 28
#define WIDTH 28

using namespace std;

vector<vector<unsigned char>> read_mnist(string);
vector<unsigned char> read_mnist_labels(string);
int reverse_int(int);

int main() {
    NeuralNetwork net (3, LENGTH * WIDTH, 30, 10);
    vector<vector<unsigned char>> training_data = read_mnist("./mnist/train-images.idx3-ubyte");
    vector<unsigned char> training_labels = read_mnist_labels("./mnist/train-labels.idx1-ubyte");

    vector<pair<vector<double>, vector<double>>> samples;
    for (int i = 0; i < training_data.size(); ++i) {
        vector<double> input, output;
        
        for (int j = 0; j < training_data[i].size(); ++j) {
            input.push_back((double) training_data[i][j]);
        }

        for (int j = 0; j < 10; ++j) {
            output.push_back(0.0);
        }
        output[(int) training_labels[i]] = 1.0;

        samples.push_back(make_pair(input, output));
    }

    net.train(samples, 1, 1, 0.7, 0);

    vector<vector<unsigned char>> test_data = read_mnist("./mnist/t10k-images.idx3-ubyte");
    vector<unsigned char> test_labels = read_mnist_labels("./mnist/t10k-labels.idx1-ubyte");

    int correct_count = 0;
    int got;
    double net_val;
    for (int i = 0; i < test_data.size(); ++i) {
       vector<double> input;
       for (int j = 0; j < test_data[i].size(); ++j) {
           input.push_back((double) test_data[i][j]);
       }

       vector<double> net_vector = net.feedforward(input);

       got = 0;
       net_val = net_vector[0];
       for (int j = 1; j < net_vector.size(); ++j) {
           if (net_vector[j] > net_val) {
               got = j;
               net_val = net_vector[j];
           }
       }

       if (got == (int) test_labels[i]) {
           ++correct_count;
       }
    }

    cout << "Accuracy: " << (((double) correct_count) / test_data.size()) << endl;
    return 0;
}

vector<vector<unsigned char>> read_mnist(string path) {
    ifstream file (path);
    
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char *) &magic_number, sizeof(magic_number)); 
        magic_number = reverse_int(magic_number);

        file.read((char *) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        
        file.read((char *)&n_rows,sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        
        file.read((char *) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        
        vector<vector<unsigned char>> ret;
        for (int i = 0; i < number_of_images; ++i) {
            vector<unsigned char> image;

            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char *) &temp, sizeof(temp));

                    image.push_back(temp);
                }
            }

            ret.push_back(image);
        }

        return ret;
    }
}

vector<unsigned char> read_mnist_labels(string full_path) {
    ifstream file(full_path);

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;

        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if (magic_number != 2049) {
            throw runtime_error("Invalid MNIST label file!");
        }

        file.read((char *) &number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverse_int(number_of_labels);
        
        vector<unsigned char> dataset;
        for(int i = 0; i < number_of_labels; ++i) {
            unsigned char temp = 0;
            file.read((char *) &temp, sizeof(temp));
            
            dataset.push_back(temp);    
        }
                                    
        return dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");    
    }
}

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}
