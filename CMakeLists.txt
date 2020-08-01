cmake_minimum_required(VERSION 3.16)
project(v2)

set(CMAKE_CXX_STANDARD 20)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include") 

add_executable(
    Perceptron
    include/NeuralNetwork/Layer.hpp
    include/NeuralNetwork/Neuron.hpp
    include/NeuralNetwork/NeuralNetwork.hpp
    include/Utils/Utils.hpp
    include/Utils/Stopwatch.hpp
    src/Utils/Utils.cpp
    src/Utils/Stopwatch.cpp
    src/NeuralNetwork/Neuron.cpp
    src/NeuralNetwork/Layer.cpp
    src/NeuralNetwork/NeuralNetwork.cpp
    main.cpp
        include/Data/iris_data.hpp include/Data/iris_dataset.hpp src/Data/iris_dataset.cpp)