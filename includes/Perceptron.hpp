//
// Created by Mauricio on 7/5/2020.
//

#ifndef PERCEPTRON_PERCEPTRON_HPP
#define PERCEPTRON_PERCEPTRON_HPP

#include <vector>

#include "Layer.hpp"

class Perceptron {
public:
    Perceptron(const std::vector<int> &topology);

    void initialize_network();
    void train(int epochs);
    void set_inputs(const std::vector<std::vector<double>> &inputs);
    void set_targets(const std::vector<double> &targets);
    void set_learning_rate(double learning_rate);
    void randomize_weights();
    void set_bias(std::vector<double> bias);
    void print_network();
    std::vector<double> get_error_history();
    std::vector<double> predict(const std::vector<double> &inputs);

private:
    std::vector<int> topology;
    std::vector<double> error_history;
    std::vector<double> targets;
    std::vector<double> bias;
    std::vector<std::vector<double>> inputs;
    std::vector<Layer*> layers;
    std::vector<Matrix*> weights_matrix;
    double learning_rate;
    double error_rate;

};


#endif //PERCEPTRON_PERCEPTRON_HPP
