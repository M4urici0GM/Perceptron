//
// Created by Mauricio on 7/5/2020.
//

#ifndef PERCEPTRON_MATRIX_HPP
#define PERCEPTRON_MATRIX_HPP

#include <vector>

class Matrix {
public:
    Matrix(int rows, int cols, bool randomize);
    Matrix(std::vector<std::vector<double>> values);

    double get_value(int row, int col);
    void set_value(int row, int col, double value);
    void set_values(std::vector<std::vector<double>> values);
    std::vector<double> get_row(int index);
    int get_cols();
    int get_rows();
    void print_matrix();
    Matrix* transpose();

private:
    int rows;
    int cols;
    std::vector<std::vector<double>> values;
};

#endif //PERCEPTRON_MATRIX_HPP
