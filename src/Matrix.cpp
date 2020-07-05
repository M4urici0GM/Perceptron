//
// Created by Mauricio on 7/5/2020.
//

#include <iostream>

#include "../includes/Matrix.hpp"
#include "../includes/Utils.hpp"

Matrix::Matrix(int rows, int cols, bool randomize) {
    this->rows = rows;
    this->cols = cols;
    this->values.clear();
    for (int i = 0; i < rows; i++) {
        std::vector<double> row;
        for (int j = 0; j < cols; j++) {
            double value = (randomize) ? Utils::generate_number() : 0.00;
            row.push_back(value);
        }
        this->values.push_back(row);
    }
}

int Matrix::get_cols() { return this->cols; }
int Matrix::get_rows() { return this->rows; }
double Matrix::get_value(int row, int col) { return this->values.at(row).at(col); }
void Matrix::set_value(int row, int col, double value) { this->values.at(row).at(col) = value; }

void Matrix::print_matrix() {
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            printf("%f \t", this->values.at(i).at(j));
        }
        std::cout << std::endl;
    }
}