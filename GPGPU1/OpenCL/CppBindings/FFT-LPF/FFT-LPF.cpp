#include <CL/cl2.hpp>
#include <clFFT.h>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE

void checkerr(cl_int err, const char* name)
{
    if (err != CL_SUCCESS)
    {
        throw cl::Error{ err, name };
    }
}


int main()
{
    try
    {
        cl::CommandQueue queue = cl::CommandQueue::getDefault();

        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        std::cout << "Default queue on platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "Default queue on device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Init computation
        const std::size_t N = std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
                                                              //     promises no data is lost, silences compiler warning
        std::vector<cl_float> vec_x(N),
                              vec_y(N);

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ -1.0f, 1.0f }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec_x), N, prng);

        cl::Buffer buf_x{ context, std::begin(vec_x), std::end(vec_x), false },
                   buf_y{ context, std::begin(vec_y), std::end(vec_y), false };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec_x), std::end(vec_x), buf_x);
        cl::copy(queue, std::begin(vec_y), std::end(vec_y), buf_y);

        // clFFT initialization
        clfftSetupData fftSetup;
        checkerr(clfftInitSetupData(&fftSetup), "clfftInitSetupData");
        checkerr(clfftSetup(&fftSetup), "clffftSetup");

        clfftPlanHandle plan;
        checkerr(clfftCreateDefaultPlan(&plan, context(), CLFFT_1D, &N), "clCreateDefaultPlan");

        checkerr(clfftSetPlanBatchSize(plan, 1), "clfftSetPlanBatchSize");
        checkerr(clfftSetPlanPrecision(plan, CLFFT_SINGLE), "clfftSetPlanPrecision");
        checkerr(clfftSetLayout(plan, CLFFT_REAL, CLFFT_COMPLEX_INTERLEAVED), "clfftSetLayout");
        checkerr(clfftSetResultLocation(plan, CLFFT_INPLACE), "clfftSetResultLocation");

        // Bake plan
        checkerr(clfftBakePlan(plan, 1u, &queue(), nullptr, nullptr), "clfftBakePlan");

        // Launch kernels
        cl::Event forward_fft_event;
        checkerr(
            clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue(), 0, nullptr, &forward_fft_event(), &buf_x(), &buf_y(), nullptr),
            "clfftBakePlan"
        );

        cl::finish();
    }
    catch (cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

        std::exit(error.err());
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
