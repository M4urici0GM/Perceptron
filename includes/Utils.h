//
// Created by Mauricio on 7/5/2020.
//

#ifndef PERCEPTRON_UTILS_H
#define PERCEPTRON_UTILS_H

class Utils {
public:
    static double generate_number();
    static double sigmoid(double x);
    static int sign(double x);
    static std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix);
    static std::vector<std::vector<double>> multiply_matrix(std::vector<std::vector<double>> matrixA, std::vector<std::vector<double>> matrixB);
    static std::vector<std::vector<double>> multiply_scalar(std::vector<std::vector<double>> matrix, double scalar);
};

#endif //PERCEPTRON_UTILS_H
