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
    static Matrix* subtract_matrix(Matrix* matrixA, Matrix* matrixB);
    static Matrix* multiply_scalar(Matrix* matrix, double scalar);
    static Matrix* hadamard_product(Matrix* matrixA, Matrix* matrixB);
    static Matrix* sum(Matrix* matrixA, Matrix* matrixB);
};

#endif //PERCEPTRON_UTILS_HPP
