//
// Created by Mauricio on 7/5/2020.
//

#ifndef PERCEPTRON_NEURON_HPP
#define PERCEPTRON_NEURON_HPP

class Neuron {
public:
    Neuron(double value);

    double get_value();
    double get_activated_value();
    void set_value(double new_value);
    void activate();
private:
    double value;
    double activated_value;
};

#endif //PERCEPTRON_NEURON_HPP
