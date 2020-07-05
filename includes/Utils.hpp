//
// Created by Mauricio on 7/5/2020.
//

#ifndef PERCEPTRON_UTILS_HPP
#define PERCEPTRON_UTILS_HPP

#include <vector>

#include "Matrix.hpp"

class Utils {
public:
    static double generate_number();
    static double sigmoid(double x);
    static int sign(double x);

    static Matrix* multiply_matrix(Matrix* matrixA, Matrix* matrixB);
    static std::vector<std::vector<double>> multiply_scalar(std::vector<std::vector<double>> matrix, double scalar);
};

#endif //PERCEPTRON_UTILS_HPP
