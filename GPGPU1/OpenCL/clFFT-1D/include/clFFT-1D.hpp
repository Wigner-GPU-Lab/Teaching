#pragma once

// Standard C++ includes
#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <future>
#include <iterator>
#include <complex>
#include <random>
#include <chrono>

// OpenCL C++ includes
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// clFFT includes
#include <clFFT.h>


namespace util
{
    template <cl_int From, cl_int To, typename Dur = std::chrono::nanoseconds>
    auto get_duration(cl::Event& ev)
    {
        return std::chrono::duration_cast<Dur>(std::chrono::nanoseconds{ ev.getProfilingInfo<To>() - ev.getProfilingInfo<From>() });
    }
}

void checkerr(cl_int err, const char* name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << name << "(" << err << ")" << std::endl;
		exit(err);
	}
}

namespace cl
{
	namespace fft
	{
		clfftStatus	bakePlan(clfftPlanHandle plHandle, cl::CommandQueue& commQueueFFT)
		{
			return clfftBakePlan(plHandle, 1u, &commQueueFFT(), nullptr, nullptr);
		}

		clfftStatus enqueueTransform(clfftPlanHandle& plHandle,
									 clfftDirection dir,
									 cl::CommandQueue& queue,
									 const std::vector<cl::Event>& waitEvents,
									 cl::Event& outEvent,
									 cl::Buffer& inputBuffer,
									 cl::Buffer& outputBuffer = cl::Buffer(),
									 cl::Buffer& tmpBuffer = cl::Buffer())
		{
			std::vector<cl_event> cl_waitEvents(waitEvents.size());
			std::transform(waitEvents.cbegin(), waitEvents.cend(), std::back_inserter(cl_waitEvents), [](const cl::Event& evnt) {return evnt();});

			return clfftEnqueueTransform(plHandle,
										 dir,
										 1,
										 &queue(),
										 //static_cast<cl_uint>(waitEvents.size()),
										 //reinterpret_cast<const cl_event*>(waitEvents.data()), // Here we make use of the fact that in memory, an array of cl_event and cl::Event are identical.
										 static_cast<cl_uint>(cl_waitEvents.size()),
										 cl_waitEvents.data(),
										 &outEvent(),
										 &inputBuffer(),
										 outputBuffer() != cl::Buffer()() ? &outputBuffer() : nullptr,
										 tmpBuffer() != cl::Buffer()() ? tmpBuffer() : static_cast<cl_mem>(NULL));
		}
	}
} // namescpace cl
