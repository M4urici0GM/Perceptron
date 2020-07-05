//
// Created by Mauricio on 7/5/2020.
//

#include <random>
#include <ctime>
#include <cmath>
#include <iostream>

#include "../includes/Utils.hpp"
#include "../includes/Matrix.hpp"

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


Matrix* Utils::multiply_matrix(Matrix *matrixA, Matrix *matrixB) {
    if (matrixA->get_cols() != matrixB->get_rows()) {
        std::cout << "Numer of cols differs to the number of rows of the second matrix." << std::endl;
        exit(1);
    }
    auto *newMatrix = new Matrix(matrixA->get_rows(), matrixB->get_cols(), false);

    for (int i = 0; i < matrixA->get_rows(); i++) {
        for (int j = 0; j < matrixB->get_cols(); j++) {
            for (int k = 0; k < matrixA->get_rows(); k++) {
                double p = matrixA->get_value(i, k) * matrixB->get_value(k, j);
                double newVal = newMatrix->get_value(i, j) + p;
                newMatrix->set_value(i, j, newVal);
            }
        }
    }

    return newMatrix;
}
