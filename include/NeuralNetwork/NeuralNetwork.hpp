#pragma once

#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>

#include "Layer.hpp"

namespace OpenNN
{
    class NeuralNetwork
    {
        public:
            NeuralNetwork(std::vector<int> topology, double learning_rate);
            ~NeuralNetwork();

            Eigen::MatrixXd predict(Eigen::MatrixXd inputs);
            void initialize_network();
            void train(int epochs, Eigen::MatrixXd inputs, Eigen::MatrixXd targets);
            void print_network();
            void calculate_error(Eigen::MatrixXd output, Eigen::MatrixXd target);
            void load_model(const char* filename);
            void save_model(const char* filename);

            std::vector<double> get_errors();
        private:
            std::vector<Layer *> network_layers;
            std::vector<Eigen::MatrixXd *> weight_matrices;
            std::vector<int> topology;
            std::vector<double> historical_errors;
            double learning_rate;

            Layer* get_layer(int index);
            std::vector<Layer *> get_layers();
    };
};
