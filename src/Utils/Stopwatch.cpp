#include "../../include/Utils/Stopwatch.hpp"

#include <chrono>

Utils::Stopwatch::Stopwatch(const int unity_type)
{
    this->unity_type = unity_type;
};

void Utils::Stopwatch::start()
{
    this->start_time = std::chrono::high_resolution_clock::now();
};

void Utils::Stopwatch::stop()
{
    this->stop_time = std::chrono::high_resolution_clock::now();
};

uint64_t Utils::Stopwatch::total_time()
{
    std::chrono::duration<uint64_t, std::nano> total = (this->stop_time - this->start_time);
    switch (this->unity_type)
    {
        case Utils::MICROSECONDS:
            return std::chrono::duration_cast<std::chrono::microseconds>(total).count();
        case Utils::MILISECONDS:
            return std::chrono::duration_cast<std::chrono::milliseconds>(total).count();
        default:
            return std::chrono::duration_cast<std::chrono::nanoseconds>(total).count();
    }
    return 0;
}
