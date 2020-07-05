//
// Created by Mauricio on 7/5/2020.
//

#include <random>
#include <ctime>
#include <cmath>

#include "../includes/Utils.h"

std::mt19937 generate(std::time(nullptr));
std::uniform_real_distribution<> distribution(0, 1);

double Utils::generate_number() {
    return distribution(generate);
}

double Utils::sigmoid(double x) {
    double result =  ( x / (1 + std::abs(x)));
    return result <= 0 ? 0 : result;
}

int Utils::sign(double x) {
    return (x >= 0) ? 1 : -1;
}


std::vector<std::vector<double>> Utils::multiply_matrix(std::vector<std::vector<double>> matrixA, std::vector<std::vector<double>> matrixB) {
    std::vector<std::vector<double>> new_matrix;
    for (int i = 0; i < matrixA.at(0).size(); i++) {
        std::vector<double> row;
        for (int j = 0; j < matrixB.size(); j++) {
            row.push_back(0);
//            new_matrix.at(i).at(j) = 0;
        }
        new_matrix.push_back(row);
    }
    for (int i = 0; i < matrixA.size(); i++)
    {
        for (int j = 0; j < matrixB.at(0).size(); j++)
        {
            for (int k = 0; k < matrixA.size(); k++)
            {
                double p = matrixA.at(i).at(k) * matrixB.at(k).at(j);
                double newVal = new_matrix.at(i).at(j) + p;
                new_matrix.at(i).at(j) = newVal;
            }
        }
    }
    return new_matrix;
}

std::vector<std::vector<double>> Utils::transpose(std::vector<std::vector<double>> matrix) {
    std::vector<std::vector<double>> new_matrix;
    for (int i = 0; i < matrix.at(0).size(); i++) {
        std::vector<double> row;
        for (int j = 0; j < matrix.size(); j++) {
            row.push_back(0.00);
//            new_matrix.at(i).at(j) = 0;
        }
        new_matrix.push_back(row);
    }
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix.at(i).size(); j++) {
            new_matrix.at(j).at(i) = matrix.at(i).at(j);
        }
    }
    return new_matrix;
}
