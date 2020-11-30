#include <CL/cl2.hpp>
#define CLRNG_SINGLE_PRECISION
#include <clRNG/clRNG.h>
#include <clRNG/mrg31k3p.h>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE
#include <execution>
#include <cstdlib>      // std::getenv
#include <filesystem>

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

        // User defined input
        auto kernel_op = "unsigned int op(unsigned int lhs, unsigned int rhs) { return lhs + rhs; }";
        auto host_op = [](cl_uint lhs, cl_uint rhs){ return lhs + rhs; };
        auto zero_elem = cl_uint{0};
        const std::size_t N = std::size_t(std::pow(2u, 28u)), // 256M, cast denotes floating-to-integral conversion,
                                                              //     promises no data is lost, silences compiler warning
                          gens_per_item = 128,
                          count_length = N / gens_per_item;
        if (N % gens_per_item != 0) throw std::runtime_error{"gens_per_item must be a divisor of N."};

        // Load program source
        std::ifstream source_file{ "./MC-Pi.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./MC-Pi.cl" };

        // Create program and kernel
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} }.append(kernel_op) };
        auto clrng_root_env = std::getenv("CLRNG_ROOT");
        if (clrng_root_env == nullptr) throw std::runtime_error{ "Specify environment variable CLRNG_ROOT for kernel locations." };
        std::filesystem::path clrng_root_path{ clrng_root_env };
        if (!std::filesystem::exists(clrng_root_path)) throw std::runtime_error{ std::string{"The path specified in the env var CLRNG_ROOT (" + clrng_root_path.string() + ") doesn't exist" } };

        std::string build_options = std::string{"-I"} + (clrng_root_path / "include").string();
        program.build({ device }, build_options.c_str());

        auto count = cl::KernelFunctor<cl_uint, cl::Buffer, cl::Buffer>(program, "count");
        auto reduce = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_uint, cl_uint>(program, "reduce");
        auto wgs = reduce.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

        while (device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() < wgs * 2 * sizeof(cl_uint))
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
        auto streams = [](std::size_t n)
        {
            clrngStatus clrng_err = CLRNG_SUCCESS;
            std::size_t streamBufferSize;

            clrngMrg31k3pStream* streams_ptr = clrngMrg31k3pCreateStreams(nullptr, n, &streamBufferSize, &clrng_err);
            if(clrng_err != CLRNG_SUCCESS)
                throw std::runtime_error{"Failed to create random streams."};

            return std::vector<clrngMrg31k3pStream>(streams_ptr, streams_ptr + n);
        }(count_length);
        

        cl::Buffer streams_buf{ context, streams.begin(), streams.end(), true },
                   front{ context, CL_MEM_READ_WRITE, count_length * sizeof(cl_uint) },
                   back{ context, CL_MEM_READ_WRITE, new_size(count_length) * sizeof(cl_uint) };;

        // Explicit (blocking) dispatch of data before launch
        std::cout << "Executing..." << std::endl;
        auto dev_start = std::chrono::high_resolution_clock::now();
        cl::copy(queue, streams.begin(), streams.end(), streams_buf);

        // Launch kernels
        cl::Event count_event = count(
            cl::EnqueueArgs{
                queue,
                count_length
            },
            gens_per_item,
            streams_buf,
            front
        );
        std::vector<cl::Event> passes(1, count_event);
        cl_uint curr = static_cast<cl_uint>(count_length);
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
                    cl::Local(factor * sizeof(cl_uint)),
                    curr,
                    zero_elem
                )
            );

            curr = static_cast<cl_uint>(new_size(curr));
            if (curr > 1) std::swap(front, back);
        }
        for (auto& pass : passes) pass.wait();

        // (Blocking) fetch of results
        cl_uint dev_res;
        cl::copy(queue, back, &dev_res, &dev_res + 1);

        auto dev_end = std::chrono::high_resolution_clock::now();

        std::vector<cl_uint> counts(count_length);
        auto stream_to_count = [=](clrngMrg31k3pStream stream)
        {
            cl_uint count = 0;
            for(int i = 0 ; i < gens_per_item ; ++i)
            {
                std::array<float,2> pos = {
                    clrngMrg31k3pRandomU01(&stream),
                    clrngMrg31k3pRandomU01(&stream)
                };

                if (pos[0] * pos[0] + pos[1] * pos[1] < 1) count++;
            }

            return count;
        };

        auto par_start = std::chrono::high_resolution_clock::now();
        std::transform(std::execution::par_unseq, streams.cbegin(), streams.cend(), counts.begin(), stream_to_count);
        auto par_ref = std::reduce(std::execution::par_unseq, counts.cbegin(), counts.cend(), zero_elem, host_op);
        auto par_end = std::chrono::high_resolution_clock::now();

        auto seq_start = std::chrono::high_resolution_clock::now();
        std::transform(std::execution::seq, streams.cbegin(), streams.cend(), counts.begin(), stream_to_count);
        auto seq_ref = std::reduce(std::execution::seq, counts.cbegin(), counts.cend(), zero_elem, host_op);
        auto seq_end = std::chrono::high_resolution_clock::now();

        auto count_to_pi = [N](cl_uint inside){ return 4. * inside / N; };

        // Validate (compute saxpy on host and match results)
        if (dev_res != par_ref || dev_res != seq_ref)
        {
            std::cerr << "Sequential reference: " << seq_ref << std::endl;
            std::cerr << "Parallel reference: " << par_ref << std::endl;
            std::cerr << "Device result: " << dev_res << std::endl;
            throw std::runtime_error{ "Validation failed!" };
        }
        else
        {
            constexpr double pi = 3.1415926535897932384626433832795028841971;
            std::cout << "pi up to machine precision: " << pi << std::endl;
            std::cout << "pi up to MC precision: " << count_to_pi(dev_res) << std::endl;
            std::cout << "Device execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(dev_end - dev_start).count() << "ms." << std::endl;
            std::cout << "Serial host execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start).count() << "ms." << std::endl;
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
