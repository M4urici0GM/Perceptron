#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <ctime>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

#include "include/NeuralNetwork/NeuralNetwork.hpp"
#include "include/Utils/Utils.hpp"
#include "include/Utils/Stopwatch.hpp"


std::vector<std::string> split_string(const std::string& string, char delimiter)
{
    std::vector<std::string> splitted_strings;
    std::string data;
    std::istringstream dataStream(string);
    while (std::getline(dataStream, data, delimiter))
    {
        splitted_strings.push_back(data);
    }
    return splitted_strings;
};

class iris_data
{
    public:
        iris_data(std::vector<double> values, std::string class_name)
        {
            this->values = values;
            this->class_name = class_name;
            this->target = { 0, 0, 0 };
        }

        iris_data()
        {};

        void set_values(std::vector<double> values){ this->values = values; }
        void set_class_name(std::string class_name) { this->class_name = class_name; }

        std::string get_class_name() { return this->class_name; }
        std::vector<double> get_values() { return this->values; }
        std::vector<double> get_target() { return this->target; }
        void set_target_value(int index) { this->target.at(index) = 1; }

    private:
        std::vector<double> values;
        std::vector<double> target;
        std::string class_name;
};

class iris_dataset
{
    public:
        iris_dataset(std::vector<std::string> data_rows)
        {
            
            for (int i = 0; i < data_rows.size(); i++)
            {
                std::vector<double> values;
                std::vector<std::string> splitted_row = split_string(data_rows.at(i), ',');
                iris_data* iris = new iris_data({}, splitted_row.at(splitted_row.size() - 1));
                for (int j = 0; j < splitted_row.size() - 1; j++)
                {
                    std::string value = splitted_row.at(j);
                    double parsed_value = std::stod(value);
                    values.push_back(parsed_value);
                }
                std::string class_name = splitted_row.at(splitted_row.size() - 1);
                iris->set_values(values);

                if (std::find(this->unique_classes.begin(), this->unique_classes.end(), class_name) == this->unique_classes.end())
                    this->unique_classes.push_back(class_name);

                this->dataset_rows.push_back(iris);
            }
            this->initialize_targets();
        }

        void print_data()
        {
            std::cout << "Classname\tValues\tTarget" << std::endl;
            for (iris_data* iris : this->dataset_rows)
            {
                std::cout << iris->get_class_name() << "\t";

                for (double value : iris->get_values())
                    printf("%.2f,", value);

                std::cout << "\t";

                for (double target : iris->get_target())
                    std::cout << target << ", ";

                std::cout << std::endl;
            }
        }

        Eigen::MatrixXd to_train_matrix()
        {
            int total_rows = this->dataset_rows.size();
            int train_data_count = std::floor(total_rows * .10);

            Eigen::MatrixXd train_data = Eigen::MatrixXd(train_data_count, 4);

            for (int i = 0; i < train_data_count; i++)
            {
                iris_data* iris = this->dataset_rows.at(i);
                std::vector<double> input_values = iris->get_values();
                for (int j = 0; j < input_values.size(); j++)
                {
                    train_data.row(i).col(j) << input_values.at(j);
                }
            }

            return train_data;
        }

        Eigen::MatrixXd to_target_matrix()
        {
            int total_rows = this->dataset_rows.size();
            int train_data_count = std::floor(total_rows * .10);

            Eigen::MatrixXd target_data = Eigen::MatrixXd(train_data_count, 3);

            for (int i = 0; i < train_data_count; i++)
            {
                iris_data* iris = this->dataset_rows.at(i);
                std::vector<double> targets = iris->get_target();
                for (int j = 0; j < targets.size(); j++)
                {
                    target_data.row(i).col(j) << targets.at(j);
                }
            }

            return target_data;
        }


        std::vector<iris_data*> get_dataset() { return this->dataset_rows; };

    private:
        std::vector<iris_data*> dataset_rows;
        std::vector<std::string> unique_classes;

        void initialize_targets()
        {
            for (iris_data* iris : this->dataset_rows)
            {
                std::vector<std::string>::iterator search_result = std::find(this->unique_classes.begin(), this->unique_classes.end(), iris->get_class_name());
                int index = std::distance(this->unique_classes.begin(), search_result);
                iris->set_target_value(index);
            }
        }
};


int main()
{
    OpenNN::NeuralNetwork* neural_network = new OpenNN::NeuralNetwork({ 4, 5, 3 }, 0.5);

    neural_network->initialize_network();
    // neural_network->print_network();

    Eigen::MatrixXd train_data;
    Eigen::MatrixXd target_data;


    std::ifstream train_data_file ("iris.data");

    
    if (train_data_file.is_open())
    {  
        std::string line;
        std::vector<std::string> data_lines;

        while(std::getline(train_data_file, line))
            data_lines.push_back(line);

        std::random_shuffle(data_lines.begin(), data_lines.end());

        iris_dataset* dataset = new iris_dataset(data_lines);
        // dataset->print_data();

        train_data = dataset->to_train_matrix();
        target_data = dataset->to_target_matrix();

        
        neural_network->train(100, train_data, target_data);

        std::ofstream output_file ("error.data", std::ofstream::binary);

        for (double error : neural_network->get_errors())
        {
            printf("%.5f\n", error);
        }
    }
    return 0;
}