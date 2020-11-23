// OpenCL includes
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

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

static const std::size_t platformIdx = 0;
static const std::size_t deviceIdx   = 0;

static const std::size_t particle_count = 4096*2;

static const float deltat       = 4e-3f;
static const float G            = 9e-4f;//1.0f;//6.67384e-11f;

void initializeOpenCL(cl::Device& device, cl::Platform& platform)
{
    std::vector<cl::Platform> platforms;
    std::vector<std::vector<cl::Device>> devices;
    cl::Platform::get(&platforms);
    
    devices.resize(platforms.size());
    std::size_t p = 0;
    for (const auto& pl : platforms)
    {
        pl.getDevices(CL_DEVICE_TYPE_ALL, &(devices[p]));
        p += 1;
    }

    if(platformIdx >= platforms.size()           ){ std::cout << "Invalid platform index: " << platformIdx << " of " << platforms.size() << "\n"; }
    if(deviceIdx   >= devices[platformIdx].size()){ std::cout << "Invalid device index: "   << deviceIdx   << " of " << devices[platformIdx].size() << "\n"; }
    
    std::cout << "Platform: " << platforms[platformIdx].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    platform = platforms[platformIdx]; 
    device = devices[platformIdx][deviceIdx];
    std::cout << "Device: "   << device  .getInfo<CL_DEVICE_NAME>    () << std::endl;
}

cl::Program loadProgram(cl::Context& ctx, std::string const& filename)
{
    std::ifstream source_file{ filename };
    if ( !source_file.is_open() )
        throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + filename };

    std::string src{ std::istreambuf_iterator<char>{ source_file },  std::istreambuf_iterator<char>{} };

    // Create program and kernel
    return cl::Program{ ctx, src };
}