#pragma once

#include <chrono>

namespace Utils
{
    class Stopwatch
    {
        public:
            Stopwatch(const int unit_type);

            void start();
            void stop();
            
            uint64_t total_time();

        private:
            int unity_type;
            std::chrono::_V2::system_clock::time_point start_time;
            std::chrono::_V2::system_clock::time_point stop_time;
    };

    const int MILISECONDS = 1;
    const int MICROSECONDS = 2;
    const int NANOSECONDS = 3;
};