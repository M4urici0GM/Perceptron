#include "../../include/NeuralNetwork/Neuron.hpp"
#include "../../include/Utils/Utils.hpp"

OpenNN::Neuron::Neuron(double value)
{
   this->set_value(value);
}

void OpenNN::Neuron::activate()
{
    this->activated_value = Utils::sigmoid(this->value);
    this->derivated_value = Utils::derive(this->activated_value);
}

double OpenNN::Neuron::get_activated_value()
{
    return this->activated_value;
}

double OpenNN::Neuron::get_value()
{
    return this->value;
}

double OpenNN::Neuron::get_derivated_value()
{
    return this->derivated_value;
}

void OpenNN::Neuron::set_value(double value)
{
    this->value = value;
    this->activate();       
}
