#include <eigen3/Eigen/Dense>

#include "../../include/NeuralNetwork/Layer.hpp";

#include "Neuron.hpp"

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
