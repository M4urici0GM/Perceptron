#include <eigen3/Eigen/Dense>
#include <vector>

#include "../../include/NeuralNetwork/Layer.hpp"
#include "../../include/NeuralNetwork/Neuron.hpp"

OpenNN::Layer::Layer(int layer_size)
{
    for (int i = 0; i < layer_size; i++)
        this->neurons.push_back(new Neuron(0.00));
}

Eigen::MatrixXd* OpenNN::Layer::to_matrix() {
    Eigen::MatrixXd* matrix = new Eigen::MatrixXd(0, this->neurons.size());
    for (int i = 0; i < this->neurons.size(); i++) {
        matrix->operator()(0, i) = this->neurons.at(i)->get_value();
    }
    return matrix;
}

std::vector<OpenNN::Neuron *> OpenNN::Layer::get_neurons()
{
    return this->neurons;
}

Eigen::MatrixXd* OpenNN::Layer::to_matrix_activated()
{
    Eigen::MatrixXd* matrix = new Eigen::MatrixXd(0, this->neurons.size());
    for (int i = 0; i < this->neurons.size(); i++) {
        matrix->operator()(0, i) = this->neurons.at(i)->get_activated_value();
    }
    return matrix;
}
