#pragma once

#include <vector>
#include <string>

namespace Utils 
{
    double sigmoid(double value);
    double tahn(double value);
    double derive(double value);
    double random_number();
    double random_number(int start, int finish);
    std::vector<std::string> split_string(const std::string& string, char delimiter);
};