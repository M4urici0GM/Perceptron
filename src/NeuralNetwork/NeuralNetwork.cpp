
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

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

void OpenNN::NeuralNetwork::load_model(const char* filename)
{

}

void OpenNN::NeuralNetwork::save_model(const char* filename)
{

}

std::vector<double> OpenNN::NeuralNetwork::get_errors()
{
    return this->historical_errors;
}

/**
 * Return a vector of layer pointers
 * */
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


/**
 * Calculate the total network error based on output/target
 * E(Y, y) = Σ (1/2(Y - y)^2)
 * where
 *  Y = output
 *  y = target
 * */
void OpenNN::NeuralNetwork::calculate_error(Eigen::MatrixXd output, Eigen::MatrixXd target)
{

    Eigen::MatrixXd error = (output - target);

    double error_sum = 0.00;

    int col_count = error.cols();

    for (int j = 0; j < col_count; j++)
    {
        double error_value = error(0, j);
        double expValue = std::pow(error_value, 2);
        error_sum += (0.5 * expValue);
    }

    this->historical_errors.push_back(error_sum);
}

void OpenNN::NeuralNetwork::train(int epochs, Eigen::MatrixXd inputs, Eigen::MatrixXd targets)
{
    if (epochs == 0 || inputs.size() == 0 || targets.size() == 0)
    {
        std::cerr << "Missing train data!" << std::endl;
        assert(false);
    }

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < inputs.rows(); i++)
        {
            Eigen::MatrixXd input = inputs.row(i).matrix();
            Eigen::MatrixXd target = targets.row(i).matrix();

            Eigen::MatrixXd output = this->predict(input);

            Eigen::MatrixXd error = (output - target);

            this->calculate_error(output, target);

            int output_layer_index = (this->network_layers.size() - 1);
            int first_hidden_layer_index = (output_layer_index - 1);

            /**
             * Output to the firt hidden
             * ----------------------------------
             * */
            Layer* current_layer = this->get_layer(output_layer_index);
            Eigen::MatrixXd current_values = *current_layer->to_matrix_derivated();

            //Calculate the gradient of current layer
            Eigen::MatrixXd gradients = error.cwiseProduct(current_values);
            // gradients *= this->learning_rate;

            /**
             * Multiply the transposed version of the gradient matrix by the output of the previous layer 
             * and transpose it
             * */
            auto teste = *this->get_layer(first_hidden_layer_index)->to_matrix_activated();
            

            Eigen::MatrixXd delta_weights = gradients.transpose() * teste;
            Eigen::MatrixXd delta_weights_transposed = delta_weights.transpose();

            /**
             * Get the weights between the previous layer and the current one
             * */
            Eigen::MatrixXd current_weights = *this->weight_matrices.at(first_hidden_layer_index);

            /**
             * Subtract the current weights by the delta weight
             * */


            /**
             * Update the weights with the new one
             * */
            *this->weight_matrices.at(first_hidden_layer_index) = (current_weights - delta_weights_transposed);


            std::cout << "Current Weights: " << std::endl << current_weights << std::endl;
            std::cout << "Delta Weights: " << std::endl << delta_weights_transposed << std::endl;
            std::cout << "Updated Weights: " << std::endl << *this->weight_matrices.at(first_hidden_layer_index) << std::endl;
            std::cout << "=====================================" << std::endl << std::endl;
        }   
    }
}

Eigen::MatrixXd OpenNN::NeuralNetwork::predict(Eigen::MatrixXd inputs)
{

    if (inputs.rows() > 1)
    {
        std::cerr << "The number of rows cannot exceed 1" << std::endl;
        assert(false);
    }

    if (inputs.cols() != this->get_layer(0)->get_neurons().size())
    {
        std::cerr << "Invalid input size!" << std::endl;
        assert(false);
    }

    Layer* input_layer = this->get_layer(0);
    int layer_count = this->get_layers().size() - 1;

    for (int i = 0; i < inputs.size(); i++)
    {
        double value = inputs(0, i);
        input_layer->get_neuron(i)->set_value(value);
    }

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