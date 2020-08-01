#include <vector>
#include <Data/iris_dataset.hpp>
#include <Utils/Utils.hpp>

Data::iris_dataset::iris_dataset(const std::vector <std::string>& data_rows) {
    for (auto & data_row : data_rows) {
        std::vector<double> values;
        std::vector<std::string> splitString = Utils::split_string(data_row, ',');
        auto *iris = new iris_data({}, splitString.at(splitString.size() - 1));
        for (int j = 0; j < splitString.size() - 1; j++) {
            std::string value = splitString.at(j);
            double parsed_value = std::stod(value);
            values.push_back(parsed_value);
        }
        std::string class_name = splitString.at(splitString.size() - 1);
        iris->set_values(values);

        if (std::find(this->unique_classes.begin(), this->unique_classes.end(), class_name) ==
            this->unique_classes.end())
            this->unique_classes.push_back(class_name);

        this->dataset_rows.push_back(iris);
    }
    this->initialize_targets();
}

void Data::iris_dataset::print_data() {
    std::cout << "Classname\tValues\tTarget" << std::endl;
    for (iris_data *iris : this->dataset_rows) {
        std::cout << iris->get_class_name() << "\t";

        for (double value : iris->get_values())
            printf("%.2f,", value);

        std::cout << "\t";

        for (double target : iris->get_target())
            std::cout << target << ", ";

        std::cout << std::endl;
    }
}

Eigen::MatrixXd Data::iris_dataset::to_train_matrix() {
    int total_rows = this->dataset_rows.size();
    int train_data_count = std::floor(total_rows * .10);

    Eigen::MatrixXd train_data = Eigen::MatrixXd(train_data_count, 4);

    for (int i = 0; i < train_data_count; i++) {
        iris_data *iris = this->dataset_rows.at(i);
        std::vector<double> input_values = iris->get_values();
        for (int j = 0; j < input_values.size(); j++) {
            train_data.row(i).col(j) << input_values.at(j);
        }
    }

    return train_data;
}

Eigen::MatrixXd Data::iris_dataset::to_target_matrix() {
    int total_rows = this->dataset_rows.size();
    int train_data_count = std::floor(total_rows * .10);

    Eigen::MatrixXd target_data = Eigen::MatrixXd(train_data_count, 3);

    for (int i = 0; i < train_data_count; i++) {
        iris_data *iris = this->dataset_rows.at(i);
        std::vector<double> targets = iris->get_target();
        for (int j = 0; j < targets.size(); j++) {
            target_data.row(i).col(j) << targets.at(j);
        }
    }

    return target_data;
}

void Data::iris_dataset::initialize_targets() {
    for (iris_data *iris : this->dataset_rows) {
        std::vector<std::string>::iterator search_result = std::find(this->unique_classes.begin(),
                                                                     this->unique_classes.end(),
                                                                     iris->get_class_name());
        int index = std::distance(this->unique_classes.begin(), search_result);
        iris->set_target_value(index);
    }
}