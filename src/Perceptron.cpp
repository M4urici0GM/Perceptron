//
// Created by Mauricio on 7/5/2020.
//

#include <vector>
#include <cassert>
#include <iostream>

#include "../includes/Perceptron.hpp"
#include "../includes/Utils.hpp"
#include "../includes/Matrix.hpp"

Perceptron::Perceptron(const std::vector<int> &topology) {
    this->topology = topology;
    this->initialize_network();
}

void Perceptron::initialize_network() {
    this->layers.clear();
    for (int layer_size : this->topology) {
        if (layer_size == 0) {
            std::cerr << "Number of neurons cannot be 0." << std::endl;
            assert(false);
        }
        this->layers.push_back(new Layer(layer_size));
    }

    for (int i = 0; i < (this->layers.size() - 1); i++) {
        auto *current_layer = this->layers.at(i);
        auto *next_layer = this->layers.at((i + 1));
        auto *weight_matrix = new Matrix(current_layer->get_neurons().size(), next_layer->get_neurons().size(), true);
        this->weights_matrix.push_back(weight_matrix);
    }
}

void Perceptron::set_bias(std::vector<double> bias) {
    this->bias = std::move(bias);
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

std::vector<double> Perceptron::predict(const std::vector<double> &inputs) {
    if (this->layers.at(0)->get_neurons().size() != inputs.size()) {
        std::cerr << "Input size doesn't match with input layer";
        assert("false");
    }
    for (int j = 0; j < this->layers.at(0)->get_neurons().size(); j++) {
        auto* input_layer = this->layers.at(0);
        input_layer->get_neuron(j)->set_value(inputs.at(j));
    }

    for (int i = 0; i < (this->layers.size() - 1);i++) {
        auto* current_layer = this->layers.at(i);
        auto* neuron_matrix = (i == 0)
                ? current_layer->transform_to_matrix()
                : current_layer->transform_to_matrix_activated();

        auto* weight_matrix = this->weights_matrix.at(i);

        Matrix* output_matrix = Utils::multiply_matrix(neuron_matrix, weight_matrix);

        for (int j = 0; j < output_matrix->get_cols(); j++) {
            double neuron_value = output_matrix->get_value(0, j);
            this->layers.at((i + 1))
                    ->get_neuron(j)
                    ->set_value(neuron_value);
        }
    }
    auto* output_layer = this->layers.at(this->layers.size() - 1);
    std::vector<double> output_matrix;
    for (int i = 0; i < output_layer->get_neurons().size(); i++) {
        output_matrix.push_back(output_layer->get_neuron(i)->get_activated_value());
    }
    return output_matrix;
}


void Perceptron::set_learning_rate(double learning_rate) {
    this->learning_rate = learning_rate;
}

std::vector<double> Perceptron::get_error_history() {
    return this->error_history;
}

void Perceptron::train(int epochs) {

}


void Perceptron::print_network() {
    for (int i = 0; i < this->layers.size(); i++) {
        printf("Neurons in layer %i: %i\n\n", i, this->layers.at(i)->get_neurons().size());

        if (i == 0) {
            printf("Input values: \n");
            for (auto* neuron : this->layers.at(0)->get_neurons()) {
                printf("%f \t", neuron->get_value());
            }
            std::cout << std::endl << std::endl;
        }


        if (i != (this->layers.size() - 1)) {
            printf("Weights for layer: %i: \n", i);
            auto *weight_matrix = this->weights_matrix.at(i);
            weight_matrix->print_matrix();
        }

        printf("====================================\n\n");
    }
}
