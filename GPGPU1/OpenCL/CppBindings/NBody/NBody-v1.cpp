// OpenCL includes
#include <CL/cl2.hpp>

// STL includes
#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <chrono>
#include <numeric>
#include <iostream>
#include <sstream>
#include <random>

// GCC/MSVC packing macro
#ifdef __GNUC__
#define PACKED( class_to_pack ) class_to_pack __attribute__((__packed__))
#else
#define PACKED( class_to_pack ) __pragma( pack(push, 1) ) class_to_pack __pragma( pack(pop) )
#endif

#define G 6.67384e-11

struct input_particle
{
    cl_float3 pos;
    cl_float3 v;
    cl_float mass;
};

//#pragma pack(1)
struct alignas(16) particle
{
    particle() = default;
    particle(const particle&) = default;
    particle(particle&&) = default;
    ~particle() = default;

    particle& operator=(const particle&) = default;
    particle& operator=(particle&&) = default;

    particle(const input_particle& in) : mass(in.mass), pos(in.pos), v(in.v), f{ 0, 0, 0 } {}

    cl_float3 pos;
    cl_float3 v;
    cl_float3 f;
    cl_float mass;
};

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
        std::ifstream source_file{ "./NBody-v1.cl" };
        if ( !source_file.is_open() )
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./NBody-v1.cl" };

        // Create program and kernel
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} } };
        program.build({ device });

        auto interaction = cl::KernelFunctor<cl::Buffer>(program, "interaction");
        auto forward_euler = cl::KernelFunctor<cl::Buffer, float>(program, "forward_euler");

        std::vector<particle> particles;
        std::size_t particle_count = 4096;
        cl_float x_abs_range = 192.f,
                 y_abs_range = 128.f,
                 z_abs_range = 32.f,
                 mass_min = 100.f,
                 mass_max = 500.f;

        // Create block of particles
        using uni = std::uniform_real_distribution<cl_float>;
        std::generate_n(
            std::back_inserter(particles),
            particle_count,
            [prng = std::default_random_engine(),
             x_dist = uni(-x_abs_range, x_abs_range),
             y_dist = uni(-y_abs_range, y_abs_range),
             z_dist = uni(-z_abs_range, z_abs_range),
             m_dist = uni(mass_min, mass_max)]() mutable
        {
            return input_particle{
                cl_float3{
                    x_dist(prng),
                    y_dist(prng),
                    z_dist(prng),
                    0.f },
                0,
                m_dist(prng)
            };
        });

        cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, particles.size() * sizeof(particle), particles.data());

        // Run warm-up kernels
        interaction(cl::EnqueueArgs{ queue, cl::NDRange{ particle_count } }, buffer);
        forward_euler(cl::EnqueueArgs{ queue, cl::NDRange{ particle_count } }, buffer, 0.001f);

        cl::finish();

        auto start = std::chrono::high_resolution_clock::now();

        for (std::size_t n = 0; n < 1000; ++n)
        {
            interaction(cl::EnqueueArgs{ queue, cl::NDRange{ particle_count } }, buffer);
            forward_euler(cl::EnqueueArgs{ queue, cl::NDRange{ particle_count } }, buffer, 0.001f);
        }

        cl::finish();

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds." << std::endl;
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