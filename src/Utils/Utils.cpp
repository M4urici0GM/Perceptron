#include "../../include/Utils/Utils.hpp"



#include <cmath>
#include <ctime>
#include <random>

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
    std::random_device randomDevice;
	std::mt19937 generate(randomDevice());
	std::uniform_real_distribution<> distribution(0, 1);
	return distribution(generate);
}