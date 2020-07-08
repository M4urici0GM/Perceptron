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

Matrix::Matrix(std::vector<std::vector<double>> existing_values) {
    this->values = existing_values;
    this->rows = existing_values.size();
    this->cols = existing_values.at(0).size();
}

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

Matrix* Matrix::transpose() {
    auto* matrix = new Matrix(this->cols, this->rows, false);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            matrix->set_value(j, i, this->values.at(i).at(j));
        }
    }
    return matrix;
}

void Matrix::set_values(std::vector<std::vector<double> > new_values) { this->values = new_values; }
int Matrix::get_cols() { return this->cols; }
int Matrix::get_rows() { return this->rows; }
std::vector<double> Matrix::get_row(int index) { return this->values.at(index); }