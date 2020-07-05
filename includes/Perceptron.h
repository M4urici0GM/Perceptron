//
// Created by Mauricio on 7/5/2020.
//

#ifndef PERCEPTRON_PERCEPTRON_H
#define PERCEPTRON_PERCEPTRON_H

#include <vector>

class Perceptron {
public:
    Perceptron();
    Perceptron(std::vector<double> trained_model);

    void train(int epochs);
    double predict(const std::vector<double> &inputs);
    void set_inputs(const std::vector<std::vector<double>> &inputs);
    void set_targets(const std::vector<double> &targets);
    void set_learning_rate(double learning_rate);
    void randomize_weights();
    std::vector<double> get_error_history();

private:
    std::vector<double> error_history;
    std::vector<double> weights;
    std::vector<double> targets;
    std::vector<std::vector<double>> inputs;
    double learning_rate;
    double error_rate;
};


#endif //PERCEPTRON_PERCEPTRON_H
