cmake_minimum_required(VERSION 3.16)
project(Perceptron)

set(CMAKE_CXX_STANDARD 14)

add_executable(Perceptron main.cpp includes/Perceptron.h src/Perceptron.cpp includes/Utils.h src/Utils.cpp)