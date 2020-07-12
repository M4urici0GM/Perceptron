cmake_minimum_required(VERSION 3.16)
project(v2)

set(CMAKE_CXX_STANDARD 20)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include") 

add_executable(
    Perceptron
    include/NeuralNetwork/Layer.hpp
    include/NeuralNetwork/Neuron.hpp
    include/NeuralNetwork/NeuralNetwork.hpp
    src/Utils/Utils.cpp
    src/NeuralNetwork/NeuralNetwork.cpp
    main.cpp
)