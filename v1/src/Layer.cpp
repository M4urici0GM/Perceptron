//
// Created by Mauricio on 7/5/2020.
//

#include <vector>

#include "../includes/Layer.hpp"
#include "../includes/Matrix.hpp"

Layer::Layer(int n_neurons) {
    this->neuron_count = n_neurons;
    for (int i = 0; i < n_neurons; i++) {
        this->neurons.push_back(new Neuron(0.00));
    }
}

std::vector<Neuron*> Layer::get_neurons() { return this->neurons; }
Neuron* Layer::get_neuron(int index) { return this->neurons.at(index); }

Matrix* Layer::transform_to_matrix_activated() {
    auto* matrix = new Matrix(1,this->neurons.size(), false);
    for (int i = 0; i < this->neurons.size(); i++) {
        matrix->set_value(0, i, this->neurons.at(i)->get_activated_value());
    }
    return matrix;
}

Matrix* Layer::transform_to_matrix() {
    auto* matrix = new Matrix(1,this->neurons.size(), false);
    for (int i = 0; i < this->neurons.size(); i++) {
        matrix->set_value(0, i, this->neurons.at(i)->get_value());
    }
    return matrix;
}

Matrix* Layer::transform_to_derivated() {
    auto* matrix = new Matrix(1,this->neurons.size(), false);
    for (int i = 0; i < this->neurons.size(); i++) {
        matrix->set_value(0, i, this->neurons.at(i)->get_derivated_value());
    }
    return matrix;
}