#include <CL/cl2.hpp>
#include <clBLAS.h>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE

void cpu_matmul_naive(std::vector<float>& C, std::vector<float> const& A, std::vector<float> const& B, int N, int M, int K) 
{
    for(int n=0; n<N; ++n)
    {
        for(int m=0; m<M; ++m)
        {
            float sum = 0;
            for(int k=0; k<K; ++k)
            {
                sum += A[n*N+k] * B[k*K+m];
            }
            C[n*N+m] = sum;
        }
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

        if(clblasSetup() != clblasSuccess) throw std::runtime_error{"clBLAS failed to initialize."};

        // Init computation
        const std::size_t N = std::size_t(std::pow(2u, 9u)), // 1M, cast denotes floating-to-integral conversion,
                                                             //     promises no data is lost, silences compiler warning
                          M = 2 * N,
                          K = N / 2;
        std::vector<cl_float> vec_x(M * K),
                              vec_y(K * N),
                              vec_z(M * N),
                              vec_w(M * N);

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ -1.0f, 1.0f }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec_x), M * K, prng);
        std::generate_n(std::begin(vec_y), K * N, prng);

        cl::Buffer buf_x{ context, std::begin(vec_x), std::end(vec_x), true },
                   buf_y{ context, std::begin(vec_y), std::end(vec_y), true },
                   buf_z{ context, std::begin(vec_z), std::end(vec_z), false };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec_x), std::end(vec_x), buf_x);
        cl::copy(queue, std::begin(vec_y), std::end(vec_y), buf_y);

        // Launch kernels
        cl::Event gemm_event;
        clblasSgemm(
            clblasRowMajor,
            clblasNoTrans,
            clblasNoTrans,
            M,
            N,
            K,
            1.0f, buf_x(), 0, K,
            buf_y(), 0, N,
            1.0f, buf_z(), 0, N,
            1, &queue(), 0, NULL, &gemm_event()
        );

        cl::finish();

        cpu_matmul_naive(vec_w, vec_x, vec_y, (int)N, (int)M, (int)K);

        // (Blocking) fetch of results
        cl::copy(queue, buf_z, std::begin(vec_z), std::end(vec_z));

        // Validate (compute saxpy on host and match results)
        if (std::equal(vec_w.cbegin(), vec_w.cend(), vec_z.cbegin(), [](const cl_float& ref, const cl_float& res){ return (res - ref) < 1e-3f; }))
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
