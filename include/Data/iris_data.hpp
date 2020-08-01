#pragma once

#include <vector>
#include <string>

namespace Data {
    class iris_data {
    public:
        iris_data(std::vector<double> values, std::string class_name) {
            this->values = std::move(values);
            this->class_name = std::move(class_name);
            this->target = {0, 0, 0};
        }

        void set_values(std::vector<double> vector) { this->values = std::move(vector); }

        void set_class_name(std::string className) { this->class_name = std::move(className); }

        std::string get_class_name() { return this->class_name; }

        std::vector<double> get_values() { return this->values; }

        std::vector<double> get_target() { return this->target; }

        void set_target_value(int index) { this->target.at(index) = 1; }

    private:
        std::vector<double> values;
        std::vector<double> target;
        std::string class_name;
    };
}
