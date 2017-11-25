// OpenCL behavioral defines
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

// OpenCL include
#include <CL/cl2.hpp>

// SYCL include
#include <CL/sycl.hpp>

// C++ includes
#include <cstddef>  // std::size_t
#include <array>    // std::array
#include <vector>   // std::vector

using real = cl_float;
using real4 = cl::sycl::float4;

enum DoubleBuffer
	{
		Front = 0,
		Back = 1
	};


void NBodyStep(cl::sycl::queue sycl_queue,
               std::vector<cl::Memory>& gl_resources,
               std::array<cl::sycl::buffer<real4>, 2> pos_double_buf,
               std::array<cl::sycl::buffer<real4>, 2> vel_double_buf,
               std::size_t particle_count,
               bool fast_interop);