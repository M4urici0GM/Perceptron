#include "../../include/Utils/Utils.hpp"
#include "../../../../../../../Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.26.28801/include/sstream"

#include <cmath>
#include <ctime>
#include <random>
#include <iostream>
#include <string>
#include <cmath>

double Utils::sigmoid(double value)
{
    return (1 / (1.0 + std::exp(-value)));
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

std::vector<std::string> split_string(const std::string& string, char delimiter)
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
