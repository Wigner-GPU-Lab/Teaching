#pragma once

// Sample includes
#include <OpenCL-C++-API-config.hpp>

// OpenCL includes
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// C++ Standard includes
#include <vector>
#include <valarray>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ios>
#include <chrono>
#include <random>

// Checks weather device is DP calable or not
bool is_device_dp_capable(const cl::Device& device);

namespace util
{
    template <cl_int From, cl_int To, typename Dur = std::chrono::nanoseconds>
    auto get_duration(cl::Event& ev)
    {
        return std::chrono::duration_cast<Dur>(std::chrono::nanoseconds{ ev.getProfilingInfo<To>() - ev.getProfilingInfo<From>() });
    }
}
