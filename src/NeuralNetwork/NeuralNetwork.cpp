
#include <iostream>
#include <vector>

#include "../../include/NeuralNetwork/NeuralNetwork.hpp"
#include "../../include/Utils/Stopwatch.hpp"

OpenNN::NeuralNetwork::NeuralNetwork(std::vector<int> topology, double learning_rate)
{
    this->topology = topology;
}

OpenNN::NeuralNetwork::~NeuralNetwork()
{
    this->topology.clear();
    this->network_layers.clear();
    this->weight_matrices.clear();

    delete this;
}

void OpenNN::NeuralNetwork::initialize_network()
{
    for (int &layer_size : this->topology)
        this->network_layers.push_back(new Layer(layer_size));
    
    Utils::Stopwatch stopwatch(Utils::MICROSECONDS);

    stopwatch.start();

    for (int i = 0; i < this->network_layers.size() - 1; i++)
    {
        int current_layer_size = this->network_layers.at(i)->get_neurons().size();
        int next_layer_size = this->network_layers.at(i + 1)->get_neurons().size();

        Eigen::MatrixXd* matrix = new Eigen::MatrixXd(current_layer_size, next_layer_size);

        this->weight_matrices.push_back(matrix);
    }
    
    stopwatch.stop();

    std::cout << "Neural Network initialized successfully!" << std::endl;
    std::cout << "Total time: " << stopwatch.total_time() << "\u03BC seconds " << std::endl;
}

void OpenNN::NeuralNetwork::print_network()
{
    for (int i = 0; i < this->network_layers.size(); i++)
    {
        Layer* layer = this->network_layers.at(i);
        std::cout << layer->to_matrix();
    }
}