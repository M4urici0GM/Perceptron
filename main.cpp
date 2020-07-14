#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <ctime>

#include "include/NeuralNetwork/NeuralNetwork.hpp"
#include "include/Utils/Utils.hpp"

int main()
{
    OpenNN::NeuralNetwork* neural_network = new OpenNN::NeuralNetwork({ 768, 16, 16, 10 }, 0.5);

    std::vector<double> inputs;

    for (int i = 0; i < 768; i++)
        inputs.push_back(Utils::random_number());

    neural_network->initialize_network();
    // neural_network->print_network();

    Eigen::MatrixXd result_matrix = neural_network->predict(inputs);
    std::cout << "Output: " << std::endl;
    std::cout << result_matrix << std::endl;
}