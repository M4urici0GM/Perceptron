//
// Created by Mauricio on 7/5/2020.
//

#include <vector>
#include <cassert>
#include <iostream>

#include "../includes/Perceptron.h"
#include "../includes/Utils.h"

Perceptron::Perceptron() {
    this->learning_rate = 0;
}

Perceptron::Perceptron(std::vector<double> trained_model) {
    this->learning_rate = 0;
    this->weights = std::move(trained_model);
}

void Perceptron::set_inputs(const std::vector<std::vector<double> > &inputs) {
    if (inputs.empty() || inputs.at(0).empty()) {
        std::cerr << "Empty input!" << std::endl;
        assert(false);
    }
    this->inputs = inputs;
}

void Perceptron::set_targets(const std::vector<double> &targets) {
    if (targets.size() != this->inputs.size()) {
        std::cerr << "Target size need to be the same as inputs" << std::endl;
        assert(false);
    }
    this->targets = targets;
}

void Perceptron::randomize_weights() {
    if (!this->inputs.empty()) {
        for (int i = 0; i < this->inputs.at(0).size(); i++) {
            double random_number = Utils::generate_number();
            this->weights.push_back(random_number);
        }
    }
}

double Perceptron::predict(const std::vector<double> &inputs) {
    double output = 0.00;
    for (int i = 0; i < inputs.size(); i++) {
        output += inputs.at(i) * this->weights.at(i);
    }
    return Utils::sigmoid(output);
}


void Perceptron::set_learning_rate(double learning_rate) {
    this->learning_rate = learning_rate;
}

std::vector<double> Perceptron::get_error_history() {
    return this->error_history;
}

void Perceptron::train(int epochs) {
    if (this->weights.empty()) {
        std::cerr << "Weights not defined." << std::endl;
        assert(false);
    }

    double current_error_rate = 0.00;
    std::vector<double> outputs;
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < this->inputs.size(); j++) {
            double guess = this->predict(this->inputs.at(j));
            double error = this->targets.at(j) - guess;
            if (error <= 0)
                continue;
            this->error_history.push_back(error);
            for (int k = 0; k < this->weights.size(); k++) {
                this->weights.at(k) += error * this->inputs.at(j).at(k) * learning_rate;
            }
        }
    }
    this->error_rate = current_error_rate;
}

