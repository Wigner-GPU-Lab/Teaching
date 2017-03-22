#include <OpenCL-C++-API.hpp>

// Checks weather device is DP calable or not
bool is_device_dp_capable(const cl::Device& device)
{
    return (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp64")) ||
           (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_amd_fp64"));
}


int main()
{
    try // Any error results in program termination
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        for (const auto& platform : platforms) std::cout << "Found platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        
        // Choose platform with most DP capable devices
        auto plat = std::max_element(platforms.cbegin(), platforms.cend(), [](const cl::Platform& lhs, const cl::Platform& rhs)
        {
            auto dp_counter = [](const cl::Platform& platform)
            {
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                
                return std::count_if(devices.cbegin(), devices.cend(), is_device_dp_capable);
            };
            
            return dp_counter(lhs) < dp_counter(rhs);
        });

        if (plat != platforms.cend())
            std::cout << "Selected platform: " << plat->getInfo<CL_PLATFORM_VENDOR>() << std::endl; 
        else
            throw std::runtime_error{ "No double-precision capable device found." };
        
        // Obtain DP capable devices
        std::vector<cl::Device> devices;
        plat->getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        std::remove_if(devices.begin(), devices.end(), [](const cl::Device& dev) {return !is_device_dp_capable(dev); });

        cl::Device device = devices.at(0);

        std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        
        // Create context and queue
        std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>((*plat)()), 0 };
        cl::Context context{ devices, props.data() };

        cl::CommandQueue queue{ context, device, cl::QueueProperties::Profiling };
        
        // Load program source
        std::ifstream source_file{ kernel_location };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{"Cannot open kernel source: "} + kernel_location };
        
        // Create program and kernel
        cl::Program program{ context, std::string{ std::istreambuf_iterator<char>{ source_file },
                                                   std::istreambuf_iterator<char>{} } };
        
        program.build({ device }, "-cl-std=CL1.0"); // Any warning counts as a compilation error, simplest kernel syntax

        auto vecAdd = cl::KernelFunctor<cl_double, cl::Buffer, cl::Buffer>(program, "vecAdd");
        
        // Init computation
        const std::size_t chainlength = std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
                                                                        //     promises no data is lost, silences compiler warning
        std::valarray<cl_double> vec_x( chainlength ),
                                 vec_y( chainlength );
        cl_double a = 2.0;

        // Fill arrays with random values between 0 and 100
        auto prng = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<cl_double>{ -100.0, 100.0 }]() mutable { return distribution(engine); };

        std::generate_n(std::begin(vec_x), chainlength, prng);
        std::generate_n(std::begin(vec_y), chainlength, prng);

        cl::Buffer buf_x{ context, std::begin(vec_x), std::end(vec_x), true },
                   buf_y{ context, std::begin(vec_x), std::end(vec_x), false };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::cbegin(vec_x), std::cend(vec_x), buf_x);
        cl::copy(queue, std::cbegin(vec_x), std::cend(vec_x), buf_y);
        
        // Launch kernels
        cl::Event kernel_event{ vecAdd(cl::EnqueueArgs{ queue, cl::NDRange{ chainlength } }, a, buf_x, buf_y) };
        
        kernel_event.wait();
        
        std::cout <<
            "Device (kernel) execution took: " <<
            util::get_duration<CL_PROFILING_COMMAND_START,
                               CL_PROFILING_COMMAND_END,
                               std::chrono::microseconds>(kernel_event).count() <<
            " us." << std::endl;

        // Compute validation set on host
        auto start = std::chrono::high_resolution_clock::now();

        std::valarray<cl_double> ref = a * vec_x + vec_y;

        auto finish = std::chrono::high_resolution_clock::now();

        std::cout <<
            "Host (validation) execution took: " <<
            std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() <<
            " us." << std::endl;

        // (Blocking) fetch of results
        cl::copy(queue, buf_y, std::begin(vec_y), std::end(vec_y));

        // Validate (compute saxpy on host and match results)
        auto markers = std::mismatch(std::cbegin(vec_y), std::cend(vec_y),
                                     std::cbegin(ref), std::cend(ref));

        if (markers.first != std::cend(vec_y) || markers.second != std::cend(ref)) throw std::runtime_error{ "Validation failed." };

    }
    catch (cl::BuildError error) // If kernel failed to build
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

        std::exit( error.err() );
    }
    catch (cl::Error error) // If any OpenCL error happes
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        std::exit( error.err() );
    }
    catch (std::exception error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;

        std::exit( EXIT_FAILURE );
    }
    
    return EXIT_SUCCESS;
}
