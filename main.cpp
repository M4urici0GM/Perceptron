#include <iostream>
#include <vector>

#include "./includes/Perceptron.hpp"

int main() {


    Perceptron *perceptron = new Perceptron({4, 5, 3});

    std::vector<std::vector<double>> inputs_vector = {
            {4.6, 3.1, 1.5, 0.2},
            {5.0, 3.4, 1.5, 0.2},
            {6.6, 2.9, 4.6, 1.3},
            {7.7, 3.8, 6.7, 2.2}
    };

    std::vector<std::vector<double>> targets_vector = {
            {1, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
    };

    Matrix inputs(inputs_vector);
    Matrix targets(targets_vector);

    perceptron->train(500, inputs, targets);

    auto output = perceptron->predict({5.2,2.7,3.9,1.4});

    for (int i = 0; i < output.get_cols(); i++) {
        printf("Neuron %i: %f\n", i, output.get_value(0, i));
    }
    return 0;
}
