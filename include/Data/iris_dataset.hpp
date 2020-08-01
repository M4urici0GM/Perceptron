#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <Data/iris_data.hpp>

namespace Data {
    class iris_dataset {
    public:
        iris_dataset(const std::vector<std::string>& data_rows);

        void print_data();

        Eigen::MatrixXd to_train_matrix();

        Eigen::MatrixXd to_target_matrix();

        std::vector<iris_data *> get_dataset() { return this->dataset_rows; };

    private:
        std::vector<iris_data *> dataset_rows;
        std::vector<std::string> unique_classes;

        void initialize_targets();
    };
}