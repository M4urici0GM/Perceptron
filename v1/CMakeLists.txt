cmake_minimum_required(VERSION 3.16)
project(Perceptron)

set(CMAKE_CXX_STANDARD 14)

add_executable(Perceptron main.cpp includes/Perceptron.hpp src/Perceptron.cpp includes/Utils.hpp src/Utils.cpp includes/Layer.hpp includes/Neuron.hpp includes/Matrix.hpp src/Layer.cpp src/Neuron.cpp src/Matrix.cpp)