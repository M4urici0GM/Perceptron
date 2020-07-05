#include <iostream>
#include <vector>

#include "./includes/Perceptron.hpp"

int main() {


    Perceptron* perceptron = new Perceptron({ 4, 5, 3 });


    std::vector<double> output = perceptron->predict({ 4.8,3.0,1.4,0.3 });

    perceptron->print_network();

    for (int i = 0; i < output.size(); i++) {
        printf("Neuron %i: %f\n", i, output.at(i));
    }
    return 0;
}
