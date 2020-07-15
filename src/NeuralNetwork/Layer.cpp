#include <eigen3/Eigen/Dense>
#include <vector>
#include <cassert>
#include <iostream>

#include "../../include/NeuralNetwork/Layer.hpp"
#include "../../include/NeuralNetwork/Neuron.hpp"

OpenNN::Layer::Layer(int layer_size)
{
    for (int i = 0; i < layer_size; i++)
        this->neurons.push_back(new Neuron(0.00));
};

OpenNN::Neuron* OpenNN::Layer::get_neuron(int index)
{
    if (index < 0 || index >= this->neurons.size())
    {
        std::cerr << "Invalid neuron index!" << std::endl;
        assert(false);
    }
    return this->neurons.at(index);
}

Eigen::MatrixXd* OpenNN::Layer::to_matrix() {
    Eigen::MatrixXd* matrix = new Eigen::MatrixXd(1, this->neurons.size());
    for (int i = 0; i < this->neurons.size(); i++) {
        (*matrix)(0, i) = this->neurons.at(i)->get_value();
    }
    return matrix;
}

std::vector<OpenNN::Neuron *> OpenNN::Layer::get_neurons()
{
    return this->neurons;
}

Eigen::MatrixXd* OpenNN::Layer::to_matrix_activated()
{
    Eigen::MatrixXd* matrix = new Eigen::MatrixXd(1, this->neurons.size());
    for (int i = 0; i < this->neurons.size(); i++) {
        (*matrix)(0, i) = this->neurons.at(i)->get_activated_value();
    }
    return matrix;
}


Eigen::MatrixXd* OpenNN::Layer::to_matrix_derivated()
{
    Eigen::MatrixXd* matrix = new Eigen::MatrixXd(1, this->neurons.size());
    for (int i = 0; i < this->neurons.size(); i++) {
        (*matrix)(0, i) = this->neurons.at(i)->get_derivated_value();
    }
    return matrix;
}
