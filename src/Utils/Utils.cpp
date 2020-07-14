#include "../../include/Utils/Utils.hpp"



#include <cmath>
#include <ctime>
#include <random>

std::mt19937 generate(std::time(nullptr));
std::uniform_real_distribution<> distribution(-1, 1);


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

double Utils::random_number()
{
    return distribution(generate);
}