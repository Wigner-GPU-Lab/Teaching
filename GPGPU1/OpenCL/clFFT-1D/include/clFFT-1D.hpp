#ifndef HEADER_HPP
#define HEADER_HPP

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
#ifdef WIN32
#pragma warning(disable : 4996) // Disable warning for usage of deprecated functions
#endif
#include <CL/cl.hpp>


// clFFT includes
#include <clFFT.h>

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
	template <cl_bitfield From, cl_bitfield To>
	std::chrono::nanoseconds event_time_elapsed(const cl::Event& evnt)
	{
		cl_int error;
		cl_ulong from, to;

		from = evnt.getProfilingInfo<From>(&error); checkerr(error, "cl::Event::getProfilingInfo(from)");
		to = evnt.getProfilingInfo<To>(&error); checkerr(error, "cl::Event::getProfilingInfo(to)");

		return std::chrono::nanoseconds(to - from);
	}

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

class Workflow
{
public:

	Workflow() : events(4, cl::Event()) {}
	Workflow(const Workflow&) = default;
	Workflow(Workflow&&) = default;
	~Workflow() = default;

	enum Events
	{
		Write = 0,
		Migrate = 1,
		Exec = 2,
		Read = 3
	};

	std::vector<cl::Event> events;
};

template <Workflow::Events Event, cl_bitfield From, cl_bitfield To>
struct workflow_time_comparator
{
	bool operator()(const Workflow& lhs, const Workflow& rhs)
	{
		return cl::event_time_elapsed<From, To>(lhs.events.at(Event)) < cl::event_time_elapsed<From, To>(rhs.events.at(Event));
	}
};

template <Workflow::Events Event, cl_bitfield From, cl_bitfield To>
void report_workflow_stage(const std::string& message, const std::vector<Workflow>& workflows)
{
	auto max = std::max_element(workflows.cbegin(), workflows.cend(), workflow_time_comparator<Event, From, To>());
	std::cout << message << " = " << std::chrono::duration_cast<std::chrono::milliseconds>(cl::event_time_elapsed<From, To>(max->events.at(Event))).count() << " milliseconds.\n" << std::endl;
	for (auto& flow : workflows)
		std::cout << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(cl::event_time_elapsed<From, To>(flow.events.at(Event))).count() << " milliseconds." << std::endl;
	std::cout << std::endl;
}

#endif // HEADER_HPP