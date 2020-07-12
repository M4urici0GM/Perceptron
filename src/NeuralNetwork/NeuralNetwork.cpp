
#include <iostream>
#include <vector>

#include "../../include/NeuralNetwork/NeuralNetwork.hpp"

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
    
    std::cout << "Neural Network initialized successfully!" << std::endl;
}