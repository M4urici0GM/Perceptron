#pragma once

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

#include "Neuron.hpp"

namespace OpenNN
{
    class Layer
    {
        public:
            Layer(int layer_size);
            ~Layer();

            Eigen::MatrixXd* to_matrix();
            Eigen::MatrixXd* to_matrix_activated();
            Eigen::MatrixXd* to_matrix_derivated();

            int count();
            std::vector<Neuron *> get_neurons();
        private:
            int layer_size();
            std::vector<Neuron *> neurons;
    };
}