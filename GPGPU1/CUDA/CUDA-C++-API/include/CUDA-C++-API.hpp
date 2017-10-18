#pragma once

// CUDA includes
#include <cuda.h>

// Standard C++ includes
#include <iostream>   // std::cout
#include <algorithm>  // std::mismatch
#include <cmath>      // std::pow
#include <valarray>   // std::valarray
#include <ios>
#include <random>     // std::default_random_engine, etc.

namespace cuda
{
    // Standalone lambdas cannot be entrypoints to kernels. The triple-bracket
    // kernel launch operator requires a freestanding or member function to
    // operate on, which in turn can be an entry point (__global__).
    template <typename F> 
    __global__ void launch(F f) { f(); }
}
