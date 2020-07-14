#include "../../include/Utils/Utils.hpp"

#include <cmath>

double Utils::sigmoid(double value)
{
    return (value / (1.0 + value));
}

double Utils::tahn(double value)
{
    return (std::exp(value)/1 + std::exp(value));
}

double Utils::derive(double value)
{
    return (value * (1 - value));
}