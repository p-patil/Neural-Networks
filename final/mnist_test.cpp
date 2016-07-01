#include <iostream>
#include <fstream>
#include <vector>
#include "neural_network.h"

using namespace std;

vector<vector<char>> read_mnist(string);
int reverse_int(int);

int main() {
    NeuralNetwork net (3, 28 * 28, 30, 10);
    vector<vector<char>> training_data = read_mnist("./mnist/train-images.idx3-ubyte");

    for (vector<char> v : training_data) {
        for (char c : v) {
            if (((int) c) < 0 || ((int) c) > 255) {
                cout << ((int) c) << endl;
            }
        }
    }

    return 0;
}

vector<vector<char>> read_mnist(string path) {
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
        
        vector<vector<char>> ret;
        for (int i = 0; i < number_of_images; ++i) {
            vector<char> image;

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

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}
