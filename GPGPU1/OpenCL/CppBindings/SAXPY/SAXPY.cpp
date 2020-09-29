#include <CL/cl2.hpp>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE

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

        // Load program source
        std::ifstream source_file{ "./SAXPY.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./SAXPY.cl" };

        // Create program and kernel
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });

        auto saxpy = cl::KernelFunctor<cl_float, cl::Buffer, cl::Buffer>(program, "saxpy");

        // Init computation
        const std::size_t chainlength = std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
                                                                        //     promises no data is lost, silences compiler warning
        std::vector<cl_float> vec_x(chainlength),
                              vec_y(chainlength);
        cl_float a = 2.0;

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ -100.0, 100.0 }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec_x), chainlength, prng);
        std::generate_n(std::begin(vec_y), chainlength, prng);

        cl::Buffer buf_x{ std::begin(vec_x), std::end(vec_x), true },
                   buf_y{ std::begin(vec_y), std::end(vec_y), false };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec_x), std::end(vec_x), buf_x);
        cl::copy(queue, std::begin(vec_y), std::end(vec_y), buf_y);

        // Launch kernels
        saxpy(cl::EnqueueArgs{ queue, cl::NDRange{ chainlength } }, a, buf_x, buf_y);

        cl::finish();

        std::transform(
            vec_x.cbegin(),
            vec_x.cbegin(),
            vec_y.cbegin(),
            vec_x.begin(),
            [=](const cl_float& x, const cl_float& y){ return a * x + y; }
        );

        // (Blocking) fetch of results
        cl::copy(queue, buf_y, std::begin(vec_y), std::end(vec_y));

        // Validate (compute saxpy on host and match results)
        if (std::equal(vec_x.cbegin(), vec_x.cend(), vec_y.cbegin()))
            throw std::runtime_error{ "Validation failed." };

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
