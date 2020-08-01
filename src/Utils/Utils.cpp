#include "../../include/Utils/Utils.hpp"

#include <cmath>
#include <ctime>
#include <random>
#include <iostream>
#include <string>
#include <cmath>
#include <sstream>

double Utils::sigmoid(double value)
{
    return (1 / (1.0 + std::exp(value)));
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
	std::uniform_real_distribution<> distribution(-1, 1);
	return distribution(generate);
}

std::vector<std::string> Utils::split_string(const std::string& string, char delimiter)
{
    std::vector<std::string> splitted_strings;
    std::string data;
    std::istringstream dataStream(string);
    while (std::getline(dataStream, data, delimiter))
    {
        splitted_strings.push_back(data);
    }
    return splitted_strings;
};
