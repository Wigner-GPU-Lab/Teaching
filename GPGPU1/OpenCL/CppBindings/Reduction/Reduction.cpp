#include <CL/cl2.hpp>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE
#include <execution>

int main(int, char* argv[])
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
        std::ifstream source_file{ "./Reduction.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./Reduction.cl" };

        // Create program and kernel
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });

        auto reduce = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_uint, cl_float>(program, "reduce");
        auto wgs = reduce.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

        while (device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() < wgs * 2 * sizeof(cl_float))
            wgs -= reduce.getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

        if (wgs == 0) throw std::runtime_error{"Not enough local memory to serve a single sub-group."};

        auto factor = wgs * 2;
        // Every pass reduces input length by 'factor'.
        // If actual size is not divisible by factor,
        // an extra output element is produced using some
        // number of zero_elem inputs.
        auto new_size = [factor](const std::size_t actual)
        {
            return actual / factor + (actual % factor == 0 ? 0 : 1);
        };
        // NOTE: because one work-group produces one output
        //       new_size == number_of_work_groups
        auto global = [=](const std::size_t actual){ return new_size(actual) * wgs; };

        // Init computation
        const std::size_t chainlength = std::atoi(argv[1]);//std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
                                                                        //     promises no data is lost, silences compiler warning
        std::vector<cl_float> vec(chainlength);

        // Fill arrays with random values between 0 and FLOAT_MAX
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ 0, 1000.0f }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec), chainlength, prng);

        cl::Buffer front{ context, std::begin(vec), std::end(vec), false },
                   back{ context, CL_MEM_READ_WRITE, new_size(vec.size()) * sizeof(cl_float) };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec), std::end(vec), front);

        // Launch kernels
        auto dev_start = std::chrono::high_resolution_clock::now();
        std::vector<cl::Event> passes;
        cl_uint curr = static_cast<cl_uint>(vec.size());
        while ( curr > 1 )
        {
            passes.push_back(
                reduce(
                    cl::EnqueueArgs{
                        queue,
                        passes,
                        global(curr),
                        wgs
                    },
                    front,
                    back,
                    cl::Local(factor * sizeof(cl_float)),
                    curr,
                    std::numeric_limits<cl_float>().max()
                )
            );

            curr = static_cast<cl_uint>(new_size(curr));
            if (curr > 1) std::swap(front, back);
        }
        for (auto& pass : passes) pass.wait();
        auto dev_end = std::chrono::high_resolution_clock::now();

        auto serial_start = std::chrono::high_resolution_clock::now();
        auto ref = *std::min_element(vec.cbegin(), vec.cend());
        auto serial_end = std::chrono::high_resolution_clock::now();

        auto par_start = std::chrono::high_resolution_clock::now();
        auto ref2 = *std::min_element(std::execution::par_unseq, vec.cbegin(), vec.cend());
        auto par_end = std::chrono::high_resolution_clock::now();

        // (Blocking) fetch of results
        cl_float res;
        cl::copy(queue, back, &res, &res + 1);

        // Validate (compute saxpy on host and match results)
        if (ref != res || ref2 != res)
        {
            std::cerr << "Ref: " << ref << "\nRes: " << res << std::endl;
            throw std::runtime_error{ "Validation failed!" };
        }
        else
        {
            std::cout << "Device execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(dev_end - dev_start).count() << "ms." << std::endl;
            std::cout << "Serial host execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(serial_end - serial_start).count() << "ms." << std::endl;
            std::cout << "Parallel host execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(par_end - par_start).count() << "ms." << std::endl;
        }
        
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
