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
            std::vector<Eigen::MatrixXd *> train(int epochs, Eigen::MatrixXd inputs, Eigen::MatrixXd targets);
            Eigen::MatrixXd predict(Eigen::MatrixXd inputs);
            Layer* get_layer(int index);
            void print_network();
            void calculate_error(Eigen::MatrixXd output, Eigen::MatrixXd target);
        private:
            std::vector<Layer *> network_layers;
            std::vector<Eigen::MatrixXd *> weight_matrices;
            std::vector<int> topology;
            std::vector<double> historical_errors;
            double learning_rate;
    };
};
