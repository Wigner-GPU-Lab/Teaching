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
        std::ifstream source_file{ "./Reduction.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./Reduction.cl" };

        // Create program and kernel
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });

        auto reduce = cl::KernelFunctor<cl_uint, cl::LocalSpaceArg, cl::Buffer, cl::Buffer>(program, "reduce");
        auto wgs = reduce.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        while (device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() < wgs * 2 * sizeof(cl_float))
            wgs -= reduce.getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

        if (wgs == 0) throw std::runtime_error{"Not enough local memory to serve a single sub-group."};

        // Init computation
        const std::size_t chainlength = std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
                                                                        //     promises no data is lost, silences compiler warning
        std::vector<cl_float> vec(chainlength);

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ -100.0, 100.0 }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec), chainlength, prng);

        cl::Buffer front{ context, std::begin(vec), std::end(vec), false };
        cl::Buffer back{ context, CL_MEM_READ_WRITE, (vec.size() / (wgs * 2) + 1) * sizeof(cl_float) };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec), std::end(vec), front);

        // Launch kernels
        std::vector<cl::Event> passes;
        cl_uint curr = static_cast<cl_uint>(vec.size());
        do
        {
            std::cout << (curr % wgs != 0 ? ((curr / wgs) + 1) * wgs : curr) / 2 << std::endl;
            std::cout << (curr % (wgs * 2)) << std::endl;
            passes.push_back(
                reduce(
                    cl::EnqueueArgs{
                        queue,
                        passes,
                        (curr % wgs != 0 ? ((curr / wgs) + 1) * wgs : curr) / 2,
                        wgs
                    },
                    curr,
                    cl::Local(wgs * 2 * sizeof(cl_float)),
                    front,
                    back
                )
            );

            curr /= static_cast<cl_uint>(wgs * 2);
            std::swap(front, back);
        }
        while ( curr > 0 );
        for (auto& pass : passes) pass.wait();

        auto ref = *std::min_element(vec.cbegin(), vec.cend());

        // (Blocking) fetch of results
        std::vector<cl_float> res(1);
        cl::copy(queue, front, std::begin(res), std::end(res));

        // Validate (compute saxpy on host and match results)
        if (ref != res[0]) throw std::runtime_error{ "Validation failed!" };

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
