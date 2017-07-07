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
        class Error : public std::exception
        {
        public:

            Error(const clfftStatus status, const char* message = nullptr)
                : m_status(status)
                , m_message(message)
            {}

            ~Error() throw() {}

            virtual const char* what() const throw() override
            {
                return m_message == nullptr ? "empty" : m_message;
            }

            clfftStatus status(void) const { return m_status; }

        private:

            clfftStatus m_status;
            const char* m_message;
        };

        class Runtime
        {
        public:

            Runtime()
            {
                clfftStatus status;

                status = clfftInitSetupData(&m_data);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "clfftInitSetupData" };

                status = clfftSetup(&m_data);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "clfftInitSetup" };
            }

            Runtime(const Runtime&) = delete;
            Runtime(Runtime&&) = delete;

            ~Runtime()
            {
                clfftTeardown();
            }

        private:

            clfftSetupData m_data;
        };

        class Plan
        {
        public:

            template<typename InputIt>
            Plan(cl::Context context,
                 InputIt first,
                 InputIt last,
                 std::size_t batch,
                 clfftPrecision precision,
                 clfftLayout input_layout,
                 clfftLayout output_layout,
                 clfftResultLocation location)
                : m_context(context)
            {
                clfftStatus status;
                std::vector<std::size_t> lengths;

                std::copy(first, last, std::back_inserter(lengths));

                switch (lengths.size())
                {
                case static_cast<std::size_t>(clfftDim::CLFFT_1D) :
                    status = clfftCreateDefaultPlan(&m_handle, context(), clfftDim::CLFFT_1D, lengths.data());
                    break;
                case static_cast<std::size_t>(clfftDim::CLFFT_2D) :
                    status = clfftCreateDefaultPlan(&m_handle, context(), clfftDim::CLFFT_2D, lengths.data());
                    break;
                case static_cast<std::size_t>(clfftDim::CLFFT_3D) :
                    status = clfftCreateDefaultPlan(&m_handle, context(), clfftDim::CLFFT_3D, lengths.data());
                    break;
                default :
                    throw std::domain_error{ "Requested FFT dimensionality is not supported" };
                    break;
                }

                status = clfftSetPlanBatchSize(m_handle, batch);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::Plan::Plan with clfftSetPlanBatchSize" };

                status = clfftSetPlanPrecision(m_handle, precision);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::Plan::Plan with clfftSetPlanPrecision" };

                status = clfftSetLayout(m_handle, input_layout, output_layout);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::Plan::Plan with clfftSetLayout" };

                status = clfftSetResultLocation(m_handle, location);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::Plan::Plan with clfftSetResultLocation" };
            }

            Plan(const Plan& plan) : m_context(plan.m_context())
            {
                clfftCopyPlan(&m_handle, plan.m_context(), plan());
            }

            ~Plan()
            {
                clfftStatus status = clfftDestroyPlan(&m_handle);
                //if (status != clfftStatus::CLFFT_SUCCESS)
                //    throw cl::fft::Error{ status, "cl::fft::Plan::~Plan with clfftDestroyPlan" };
            }

            const clfftPlanHandle& operator()() const throw() { return m_handle; }

            void bake(cl::CommandQueue& queue)
            {
                clfftStatus status = clfftBakePlan(m_handle, 1u, &queue(), nullptr, nullptr);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::Plan::bake" };
            }

            template<typename InputIt>
            void bake(InputIt first, InputIt last)
            {
                std::vector<cl_command_queue> queues;
                queues.reserve(std::distance(first, last));
                std::transform(first, last, std::back_inserter(queues), [](const cl::CommandQueue& cpp_queue) { return cpp_queue(); });

                clfftStatus status = clfftBakePlan(m_handle, static_cast<cl_uint>(queues.size()), queues.data(), nullptr, nullptr);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::Plan::bake" };
            }

        private:

            clfftPlanHandle m_handle;
            cl::Context m_context;
        };

        class TransformArgs
        {
            friend cl::Event transform(TransformArgs args,
                                       cl::Buffer& input_buffer,
                                       cl::Buffer& output_buffer);

            friend cl::Event transform(TransformArgs args,
                                       cl::Buffer& input_buffer);
        public:

            TransformArgs(Plan plan,
                          cl::CommandQueue queue,
                          clfftDirection dir,
                          const std::vector<cl::Event>& wait_events,
                          cl::Buffer tmp = cl::Buffer{})
                : m_plan{ plan }
                , m_queue{ queue }
                , m_dir{ dir }
                , m_events(wait_events)
                , m_tmp{ tmp }
            {
            }

            template<typename InputIt>
            TransformArgs(Plan plan,
                          cl::CommandQueue queue,
                          clfftDirection dir,
                          InputIt first,
                          InputIt last,
                          cl::Buffer tmp = cl::Buffer{})
                : TransformArgs(plan,
                                queue,
                                dir,
                                std::vector<cl::Event>(first, last),
                                tmp)
            {
            }

            TransformArgs(Plan plan,
                          cl::CommandQueue queue,
                          clfftDirection dir,
                          cl::Buffer tmp = cl::Buffer{})
                : m_plan{ plan }
                , m_queue{ queue }
                , m_dir{ dir }
                , m_events(0)
                , m_tmp{ tmp }
            {
            }

        private:

            Plan m_plan;
            cl::CommandQueue m_queue;
            clfftDirection m_dir;
            std::vector<cl::Event> m_events;
            cl::Buffer m_tmp;
        };

        cl::Event transform(TransformArgs args,
                            cl::Buffer input_buffer,
                            cl::Buffer output_buffer)
        {
            clfftStatus status;
            cl::Event result;

            {
                clfftResultLocation loc;
                status = clfftGetResultLocation(args.m_plan(), &loc);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::transform with clfftGetResultLocation" };

                if (loc == clfftResultLocation::CLFFT_INPLACE)
                    throw std::logic_error{ "cl::fft::transform called with in-place plan and out-of-place syntax" };
            }

            std::vector<cl_event> cl_events;
            cl_events.reserve(args.m_events.size());
            std::transform(args.m_events.cbegin(), args.m_events.cend(), std::back_inserter(cl_events), [](const cl::Event& ev) { return ev(); });

            status = clfftEnqueueTransform(args.m_plan(),
                                           args.m_dir,
                                           1u,
                                           &args.m_queue(),
                                           static_cast<cl_uint>(cl_events.size()),
                                           cl_events.data(),
                                           &result(),
                                           &input_buffer(),
                                           &output_buffer(),
                                           args.m_tmp());

            if (status != clfftStatus::CLFFT_SUCCESS)
                throw cl::fft::Error{ status, "cl::fft::transform" };

            return result;
        }

        cl::Event transform(TransformArgs args,
                            cl::Buffer& input_buffer)
        {
            clfftStatus status;
            cl::Event result;

            {
                clfftResultLocation loc;
                status = clfftGetResultLocation(args.m_plan(), &loc);
                if (status != clfftStatus::CLFFT_SUCCESS)
                    throw cl::fft::Error{ status, "cl::fft::transform with clfftGetResultLocation" };

                if (loc != clfftResultLocation::CLFFT_INPLACE)
                    throw std::logic_error{ "cl::fft::transform called with out-of-place plan and in-place syntax" };
            }

            std::vector<cl_event> cl_events;
            cl_events.reserve(args.m_events.size());
            std::transform(args.m_events.cbegin(), args.m_events.cend(), std::back_inserter(cl_events), [](const cl::Event& ev) { return ev(); });

            status = clfftEnqueueTransform(args.m_plan(),
                                           args.m_dir,
                                           1u,
                                           &args.m_queue(),
                                           static_cast<cl_uint>(cl_events.size()),
                                           cl_events.data(),
                                           &result(),
                                           &input_buffer(),
                                           nullptr,
                                           args.m_tmp());

            if (status != clfftStatus::CLFFT_SUCCESS)
                throw cl::fft::Error{ status, "cl::fft::transform" };

            return result;
        }

		clfftStatus	bakePlan(clfftPlanHandle plHandle, cl::CommandQueue& commQueueFFT)
		{
            return clfftBakePlan(plHandle, 1u, &commQueueFFT(), nullptr, nullptr);
		}

		clfftStatus enqueueTransform(clfftPlanHandle plHandle,
									 clfftDirection dir,
									 cl::CommandQueue queue,
									 const std::vector<cl::Event>& waitEvents,
									 cl::Event outEvent,
									 cl::Buffer inputBuffer,
									 cl::Buffer outputBuffer = cl::Buffer(),
									 cl::Buffer tmpBuffer = cl::Buffer())
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
