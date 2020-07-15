#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <ctime>

#include "include/NeuralNetwork/NeuralNetwork.hpp"
#include "include/Utils/Utils.hpp"
#include "include/Utils/Stopwatch.hpp"

int main()
{
    OpenNN::NeuralNetwork* neural_network = new OpenNN::NeuralNetwork({ 4, 16, 16, 3 }, 0.5);

    neural_network->initialize_network();
    // neural_network->print_network();

    Eigen::MatrixXd train_data = Eigen::MatrixXd(1, 4);
    Eigen::MatrixXd target_data = Eigen::MatrixXd(1, 3);

    train_data << 0.2, 0.1, 0.3, 0.5;
    target_data << 1, 0, 0;

    Eigen::MatrixXd result_matrix = neural_network->predict(train_data);
    std::cout << "Output: " << result_matrix << std::endl;

    neural_network->train(10000, train_data, target_data);

    Eigen::MatrixXd result_after_train_matrix = neural_network->predict(train_data);
    std::cout << "Output: " << result_after_train_matrix << std::endl;
}