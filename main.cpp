#include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <vector>
#include <ctime>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <NeuralNetwork/NeuralNetwork.hpp>
#include <Utils/Utils.hpp>
#include <Utils/Stopwatch.hpp>
#include <Data/iris_dataset.hpp>

int main()
{
    auto* neural_network = new OpenNN::NeuralNetwork({ 4, 5, 3 }, 0.5);

    neural_network->initialize_network();
    // neural_network->print_network();

    Eigen::MatrixXd train_data;
    Eigen::MatrixXd target_data;


    std::ifstream train_data_file ("./iris.data");

    
    if (train_data_file.is_open())
    {  
        std::string line;
        std::vector<std::string> data_lines;

        while(std::getline(train_data_file, line))
            data_lines.push_back(line);

        std::shuffle(data_lines.begin(), data_lines.end(), std::mt19937(std::random_device()()));

        auto* dataset = new Data::iris_dataset(data_lines);
        // dataset->print_data();

        train_data = dataset->to_train_matrix();
        target_data = dataset->to_target_matrix();

        
        neural_network->train(100, train_data, target_data);

        std::ofstream output_file ("error.data", std::ofstream::binary);

        for (double error : neural_network->get_errors())
        {
            printf("%.5f\n", error);
        }
    }
    return 0;
}