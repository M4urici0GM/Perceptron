
#include <iostream>
#include <vector>
#include <cassert>

#include "../../include/NeuralNetwork/NeuralNetwork.hpp"
#include "../../include/Utils/Stopwatch.hpp"
#include "../../include/Utils/Utils.hpp"

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
std::vector<OpenNN::Layer *> OpenNN::NeuralNetwork::get_layers()
{
    return this->network_layers;
}


OpenNN::Layer* OpenNN::NeuralNetwork::get_layer(int index)
{
    if (index < 0 || index >= this->network_layers.size())
    {
        std::cerr << "Outside of layer range!" << std::endl;
        assert(false);    
    }

    return this->network_layers.at(index);
}

Eigen::MatrixXd OpenNN::NeuralNetwork::predict(std::vector<double> inputs)
{
    if (inputs.size() != this->get_layer(0)->get_neurons().size())
    {
        std::cerr << "Invalid input size!" << std::endl;
        assert(false);
    }

    Layer* input_layer = this->get_layer(0);
    int layer_count = this->get_layers().size() - 1;

    for (int i = 0; i < inputs.size(); i++)
        input_layer->get_neuron(i)->set_value(inputs.at(i));

    for (int i = 0; i < layer_count; i++)
    {
        Eigen::MatrixXd value_matrix = (i == 0)
            ? *this->get_layer(i)->to_matrix()
            : *this->get_layer(i)->to_matrix_activated();

        Eigen::MatrixXd weight_matrix = *this->weight_matrices.at(i);
        
        Eigen::MatrixXd result_matrix = value_matrix * weight_matrix;

        Layer* next_layer = this->get_layer(i + 1);
        for (int j = 0; j < next_layer->get_neurons().size(); j++)
            next_layer->get_neuron(j)->set_value(result_matrix(0, j));
    }

    Layer* output_layer = this->get_layer(layer_count);
    Eigen::MatrixXd result_matrix = *output_layer->to_matrix_activated();

    return result_matrix;
}

void OpenNN::NeuralNetwork::initialize_network()
{
    for (int &layer_size : this->topology)
        this->network_layers.push_back(new Layer(layer_size));
    
    Utils::Stopwatch stopwatch(Utils::MILISECONDS);

    stopwatch.start();

    for (int i = 0; i < this->network_layers.size() - 1; i++)
    {
        int current_layer_size = this->network_layers.at(i)->get_neurons().size();
        int next_layer_size = this->network_layers.at(i + 1)->get_neurons().size();

        Eigen::MatrixXd* matrix = new Eigen::MatrixXd(current_layer_size, next_layer_size);
        for (int j = 0; j < current_layer_size; j++)
            for (int k = 0; k < next_layer_size; k++)
            {
                double value = Utils::random_number();
                (*matrix)(j, k) = value;
            }
                
        this->weight_matrices.push_back(matrix);
    }
    
    stopwatch.stop();

    std::cout << "Neural Network initialized successfully!" << std::endl;
    std::cout << "Total time: " << stopwatch.total_time() << " milliseconds " << std::endl;
}

void OpenNN::NeuralNetwork::print_network()
{
    int layer_count = this->network_layers.size();
    for (int i = 0; i < layer_count; i++)
    {
        std::cout << "Values for layer " << (i + 1) << std::endl;
        Layer* layer = this->network_layers.at(i);
        std::cout << *layer->to_matrix() << std::endl;
        if (i != (layer_count - 1)) {
            Eigen::MatrixXd weight_matrix = *this->weight_matrices.at(i);
            std::cout << "Weight matrix for layer " << (i+1) << std::endl;
            std::cout << weight_matrix << std::endl;
        }
        std::cout << "____________________" << std::endl;
    }
}