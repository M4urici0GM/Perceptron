#include <iostream>
#include <vector>

#include "./includes/Perceptron.hpp"

int main() {


    Perceptron *perceptron = new Perceptron({4, 4, 3});

    std::vector<std::vector<double>> inputs_vector = {
            {5.4, 3.9, 1.7, 0.4},
            {6.3, 2.9, 5.6, 1.8},
            {4.4, 3.2, 1.3, 0.2},
            {6.7, 3.1, 5.6, 2.4},
            {6.4, 2.9, 4.3, 1.3},
            {6.1, 2.8, 4.0, 1.3},
            {5.1, 3.7, 1.5, 0.4},
			{5.1,3.8,1.9,0.4},
			{5.1,3.8,1.9,0.4},
    };

    std::vector<std::vector<double>> targets_vector = {
            {1, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {1, 0, 0},
			{1, 0, 0},
			{1, 0, 0}
    };

    Matrix inputs(inputs_vector);
    Matrix targets(targets_vector);

    perceptron->train(10000, inputs, targets);

    auto output = perceptron->predict({5.2,3.5,1.5,0.2});

    for (int i = 0; i < output.get_cols(); i++) {
        std::cout << output.get_value(0, i) << std::endl;
    }

//    std::vector<double> errors = perceptron->get_error_history();
//    for (double &error : errors) {
//        std::cout << error << std::endl;
//    }
    return 0;
}
