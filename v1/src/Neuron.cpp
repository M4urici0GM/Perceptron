//
// Created by Mauricio on 7/5/2020.
//

#include "../includes/Neuron.hpp"
#include "../includes/Utils.hpp"

Neuron::Neuron(double value) {
    this->value = value;
    if (value != 0) {
        this->activate();
    }
}

double Neuron::get_value() { return this->value; }
double Neuron::get_activated_value() { return this->activated_value; }
double Neuron::get_derivated_value() { return this->derivated_value; }

void Neuron::set_value(double new_value) {
    this->value = new_value;
    this->activate();
}

void Neuron::activate() {
    if (value == 0)
        return;
    this->activated_value = Utils::sigmoid(value);
    this->derivated_value = (this->activated_value * (1 - this->activated_value));
}

