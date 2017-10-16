#pragma once

// CUDA includes
#include <cuda.h>

// Standard C++ includes
#include <iostream>   // std::cout
#include <algorithm>  // std::max_element
#include <vector>     // std::vector
#include <cmath>      // std::abs

namespace cuda
{
    // Standalone lambdas cannot be entrypoints to kernels. The triple-bracket
    // kernel launch operator requires a freestanding or member function to
    // operate on, which in turn can be an entry point (__global__).
    template <typename F> 
    __global__ void launch_kernel(F f) { f(); }
}
