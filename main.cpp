#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>

#include "include/NeuralNetwork/NeuralNetwork.hpp"

int main()
{

    OpenNN::NeuralNetwork* neural_network = new OpenNN::NeuralNetwork({ 3, 2, 3}, 0.5);

    neural_network->initialize_network();
    neural_network->print_network();
}