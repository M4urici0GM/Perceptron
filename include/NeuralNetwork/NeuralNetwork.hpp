#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

#include "Layer.hpp"

namespace OpenNN
{
    class NeuralNetwork
    {
        public:
            NeuralNetwork(std::vector<int> topology, double learning_rate);
            ~NeuralNetwork();

            void initialize_network();

            std::vector<Layer *> get_layers();            
            std::vector<Eigen::MatrixXd *> train(Eigen::MatrixXd inputs, Eigen::MatrixXd targets);

            Layer *get_layer(int index);
            double predict(Eigen::MatrixXd input);

            void print_network();

        private:
            std::vector<Layer *> network_layers;
            std::vector<Eigen::MatrixXd *> weight_matrices;
            std::vector<int> topology;
            double learning_rate;
    };
};
