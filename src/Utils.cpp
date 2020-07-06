//
// Created by Mauricio on 7/5/2020.
//

#include <random>
#include <ctime>
#include <cmath>
#include <iostream>
#include <cassert>

#include "../includes/Utils.hpp"
#include "../includes/Matrix.hpp"

std::mt19937 generate(std::time(nullptr));
std::uniform_real_distribution<> distribution(0, 1);

double Utils::generate_number() {
    return distribution(generate);
}

double Utils::sigmoid(double x) {
    double result = (x / (1 + std::abs(x)));
    return result <= 0 ? 0 : result;
}

int Utils::sign(double x) {
    return (x >= 0) ? 1 : -1;
}


Matrix *Utils::multiply_matrix(Matrix *matrixA, Matrix *matrixB) {
    if (matrixA->get_cols() != matrixB->get_rows()) {
        std::cout << "Numer of cols differs to the number of rows of the second matrix." << std::endl;
        exit(1);
    }
    auto *newMatrix = new Matrix(matrixA->get_rows(), matrixB->get_cols(), false);

    for (int i = 0; i < newMatrix->get_rows(); i++) {
        for (int j = 0; j < newMatrix->get_cols(); j++) {
            double sum = 0;
            for (int k = 0; k < matrixA->get_cols(); k++) {
                sum += matrixA->get_value(i, k) * matrixB->get_value(k, j);
            }
            newMatrix->set_value(i, j, sum);
        }
    }
    return newMatrix;
}

Matrix *Utils::subtract_matrix(Matrix *matrixA, Matrix *matrixB) {
    if (matrixA->get_rows() != matrixB->get_rows() || matrixA->get_cols() != matrixB->get_cols()) {
        std::cerr << "Matrix dimensions do not match" << std::endl;
        assert(false);
    }
    auto *new_matrix = new Matrix(matrixA->get_rows(), matrixA->get_cols(), false);
    for (int i = 0; i < matrixA->get_rows(); i++) {
        for (int j = 0; j < matrixA->get_cols(); j++) {
            double new_value = (matrixA->get_value(i, j) - matrixB->get_value(i, j));
            new_matrix->set_value(i, j, new_value);
        }
    }
    return new_matrix;
}

Matrix *Utils::multiply_scalar(Matrix *matrix, double scalar) {
    auto *new_matrix = new Matrix(matrix->get_rows(), matrix->get_cols(), false);
    for (int i = 0; i < matrix->get_rows(); i++) {
        for (int j = 0; j < matrix->get_cols(); j++) {
            double new_value = (matrix->get_value(i, j) * scalar);
            new_matrix->set_value(i, j, new_value);
        }
    }
    return new_matrix;
}

Matrix *Utils::hadamard_product(Matrix *matrixA, Matrix *matrixB) {
    auto *new_matrix = new Matrix(matrixA->get_rows(), matrixA->get_cols(), false);
    for (int i = 0; i < matrixA->get_rows(); i++) {
        for (int j = 0; j < matrixA->get_cols(); j++) {
            double new_value = (matrixA->get_value(i, j) * matrixB->get_value(i, j));
            new_matrix->set_value(i, j, new_value);
        }
    }
    return new_matrix;
}

Matrix* Utils::sum(Matrix* matrixA, Matrix* matrixB) {
    auto *new_matrix = new Matrix(matrixA->get_rows(), matrixA->get_cols(), false);
    for (int i = 0; i < matrixA->get_rows(); i++) {
        for (int j = 0; j < matrixA->get_cols(); j++) {
            double new_value = (matrixA->get_value(i, j) + matrixB->get_value(i, j));
            new_matrix->set_value(i, j, new_value);
        }
    }
    return new_matrix;
}
