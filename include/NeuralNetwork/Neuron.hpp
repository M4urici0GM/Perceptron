#pragma once

namespace OpenNN
{
    class Neuron
    {
        public:
            Neuron(double initial_value);
            ~Neuron();

            double get_value();
            double get_activated_value();
            double get_derivated_value();

            void set_value(double value);

        private:
            double value;
            double activated_value;
            double derivated_value;
    };
};
