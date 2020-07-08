#include <iostream>
#include <vector>

#include "./includes/Perceptron.hpp"

int main() {


    Perceptron *perceptron = new Perceptron({2, 4, 1});

    std::vector<std::vector<double>> inputs_vector = {
            {1, 1},
            {1, 0},
            {0, 0},
            {1, 1},
    };

    std::vector<std::vector<double>> targets_vector = {
            {1}, {0}, {0}, {1}
    };

    Matrix inputs(inputs_vector);
    Matrix targets(targets_vector);

    perceptron->train(1000, inputs, targets);

    auto output = perceptron->predict({1, 0});

    for (int i = 0; i < output.get_cols(); i++) {
        std::cout << output.get_value(0, i) << std::endl;
    }

//    std::vector<double> errors = perceptron->get_error_history();
//    for (double &error : errors) {
//        std::cout << error << std::endl;
//    }
    return 0;
}
