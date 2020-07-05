#include <iostream>
#include <vector>

#include "./includes/Perceptron.h"

int main() {
    std::vector<std::vector<double>> inputs = {
            {1, 0, 1},
            {0, 1, 1},
            {1, 1, 1},
            {0, 0, 1},
            {0, 1, 0},
            {1, 1, 0},
    };

    std::vector<double> targets = {
            1, 0, 0, 1, 0, 1
    };

    Perceptron perceptron{};
    perceptron.set_learning_rate(0.5);
    perceptron.set_inputs(inputs);
    perceptron.set_targets(targets);
    perceptron.randomize_weights();

    perceptron.train(10000);


    std::vector<double> test_data = {1, 0, 0};
    double target = 0;

    double output = perceptron.predict(test_data);

    std::cout << "Perceptron guess: " << output << std::endl;

    return 0;
}
