//
// Created by Mauricio on 7/5/2020.
//

#ifndef PERCEPTRON_LAYER_HPP
#define PERCEPTRON_LAYER_HPP

#include <vector>

#include "Neuron.hpp"
#include "Matrix.hpp"

class Layer {
public:
    Layer(int n_neurons);

    std::vector<Neuron*> get_neurons();
    Neuron* get_neuron(int index);
    Matrix* transform_to_matrix();
    Matrix* transform_to_matrix_activated();
    Matrix* transform_to_derivated();
private:
    int neuron_count;
    std::vector<Neuron*> neurons;
};

#endif //PERCEPTRON_LAYER_HPP
