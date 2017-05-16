#include <clFFT-1D.hpp>

int main()
{
    // Test params
    std::size_t batch = 128;
    std::array<std::size_t, 1> lengths = { 65536 };
    cl_device_type dev_type = CL_DEVICE_TYPE_GPU;

    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (const auto& platform : platforms) std::cout << "Found platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

        // Choose first platform with the desired device type
        auto plat = std::find_if(platforms.cbegin(), platforms.cend(), [dev_type](const cl::Platform& plat)
        {
            std::vector<cl::Device> devs;

            plat.getDevices(dev_type, &devs);
            return !devs.empty();
        });

        if (plat != platforms.cend())
            std::cout << "Selected platform: " << plat->getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        else
            throw std::runtime_error{ "No device of the desired type found." };

        std::vector<cl::Device> devices;
        plat->getDevices(dev_type, &devices);

        // Select first device (no multi-device just yet)
        cl::Device device = devices.at(0);

        std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create context and queue
        std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>((*plat)()), 0 };
        cl::Context context{ devices, props.data() };

        cl::CommandQueue queue{ context, device, cl::QueueProperties::Profiling };

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_float>{ -100.0, 100.0 }]() mutable
        {
            return std::complex<cl_float>{ distribution(engine),
                                        distribution(engine) };
        };

        std::vector<std::complex<cl_float>> vec_x;
        vec_x.reserve(batch * lengths.at(0));

        std::generate_n(std::back_inserter(vec_x), batch * lengths.at(0), prng);

        cl::Buffer buf_x{ context, std::begin(vec_x), std::end(vec_x), false };

        // clFFT initialization
        std::cout << "Initializing clFFT" << std::endl;
        cl::fft::Runtime fft_runtime;
        cl::fft::Plan fft_plan{ context,
                                lengths.begin(), lengths.end(),
                                batch,
                                clfftPrecision::CLFFT_SINGLE,
                                clfftLayout::CLFFT_COMPLEX_INTERLEAVED,
                                clfftLayout::CLFFT_COMPLEX_INTERLEAVED,
                                clfftResultLocation::CLFFT_INPLACE };
        fft_plan.bake(queue);

        // Start the FFTs
        auto start = std::chrono::high_resolution_clock::now();

        // Explicit dispatch of data before launch
        std::vector<cl::Event> dispatch(1);
        queue.enqueueWriteBuffer(buf_x,
                                 CL_FALSE,
                                 0,
                                 batch * lengths.at(0) * sizeof(std::complex<cl_float>),
                                 vec_x.data(),
                                 nullptr,
                                 &dispatch.at(0));
        // Execute the plan
        std::vector<cl::Event> exec{ cl::fft::transform(cl::fft::TransformArgs{ fft_plan,
                                                                                queue,
                                                                                clfftDirection::CLFFT_FORWARD,
                                                                                dispatch },
                                                        buf_x) };

        // Initiate data fetch from devices
        cl::Event fetch;
        queue.enqueueReadBuffer(buf_x,
                                CL_FALSE,
                                0,
                                batch * lengths.at(0) * sizeof(std::complex<cl_float>),
                                vec_x.data(),
                                &exec,
                                &fetch);
        // Wait to finish
        queue.flush();
        fetch.wait();

        // End time
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Total time as measured by std::chrono::high_precision_timer =\n\n\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds.\n" << std::endl;
        std::cout << "Data dispatch as measured by cl::Event::getProfilingInfo =\n\n\t" << util::get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::milliseconds>(dispatch.at(0)).count() << " milliseconds.\n" << std::endl;
        std::cout << "FFT execution as measured by cl::Event::getProfilingInfo =\n\n\t" << util::get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::milliseconds>(exec.at(0)).count() << " milliseconds.\n" << std::endl;
        std::cout << "Data fetch as measured by cl::Event::getProfilingInfo =\n\n\t" << util::get_duration<CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END, std::chrono::milliseconds>(fetch).count() << " milliseconds.\n" << std::endl;
    }
    catch (cl::fft::Error error) // If any clFFT error happens
    {
        std::cerr << error.what() << "(" << error.status() << ")" << std::endl;

        std::exit(error.status());
    }
    catch (cl::Error error) // If any OpenCL error happens
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        std::exit(error.err());
    }
    catch (std::exception error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;

        std::exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
