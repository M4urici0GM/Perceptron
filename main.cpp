#include <iostream>
#include <random>
#include <chrono>
#include <vector>

std::mt19937 generate(std::time(nullptr));
std::uniform_real_distribution<> distribution(0, 1);

double sigmoid(double x) {
    return (x / (1 + x));
}

double randomNumber() {
    return distribution(generate);
}


double predict(const std::vector<double> &inputs, const std::vector<double> &weights) {
    double output = 0.00;
    for (int i = 0; i < inputs.size(); i++) {
        output += inputs.at(i) * weights.at(i);
    }
    return output;
}

std::vector<std::vector<double>>
train(int epochs, const std::vector<std::vector<double>> &input_data, const std::vector<double> &targets,
      const std::vector<double> &current_weights, double learning_rate) {
    std::vector<double> new_weights = current_weights;
    std::vector<double> errors;
    std::vector<std::vector<double>> trained_model;
    double error_rate = 0.00;

    for (int i = 0; i < epochs; i++) {

        for (int j = 0; j < input_data.size(); j++) {
            double output = 0.00;
            double error = 0.00;

            for (int k = 0; k < input_data.at(j).size(); k++) {
                double input_value = input_data.at(j).at(k);
                double weight = new_weights.at(k);
                output += input_value * weight;
            }

            double activated_output = sigmoid(output);

            error = (targets.at(j) - activated_output);
            error_rate += error;
            errors.push_back(error);
            output = 0.00;

        };

        for (int j = 0; j < input_data.size(); j++) {
            for (int k = 0; k < input_data.at(j).size(); k++) {
                double input_value = input_data.at(j).at(k);
                double weight = new_weights.at(k);
                double current_errror = errors.at(i);
                double new_weight = weight + current_errror * input_value;
                new_weights.at(k) = new_weight;
            }
        }
    }

    trained_model.push_back(new_weights);
    trained_model.push_back(errors);
    trained_model.push_back({error_rate});

    return trained_model;
}


int main() {
    std::vector<double> weights = {
            randomNumber(),
            randomNumber(),
            randomNumber(),
    };

    std::vector<std::vector<double>> inputs = {
            {0, 1, 1},
            {1, 0, 0},
            {0, 0, 0},
            {0, 1, 0}
    };

    std::vector<double> targets = {
            1,
            0,
            0,
            1,
    };

    double learning_rate = 0.5;
    std::vector<double> errors;
    double error_rate = 0.00;

    std::vector<std::vector<double>> trained_model =
            train(500, inputs, targets, weights, learning_rate);


    weights = trained_model.at(0);
    errors = trained_model.at(1);
    error_rate = trained_model.at(2).at(0);

    std::vector<double> test_data = {0, 1, 0};
    double target = 1;
    double new_output = sigmoid(predict(test_data, weights));
    double new_error = (target - new_output);

    std::cout << "Perceptron guess: " << new_output << std::endl;
    std::cout << "New perceptron error: " << new_error << std::endl;
    std::cout << "Error history: " << std::endl;
    for (int i = 0; i < errors.size(); i++) {
        std::cout << errors.at(i) << std::endl;
    }
    return 0;
}
