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
static const std::size_t nsteps = 1000;
static const bool runCPUReference = false;

static const float deltat       = 0.0000005f;
static const float G            = 1.0f;//6.67384e-11f;

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

    particle(const input_particle& in) : mass(in.mass), pos(in.pos), v(in.v), f{ 0, 0, 0, 0 } {}

    cl_float3 pos;
    cl_float3 v;
    cl_float3 f;
    cl_float mass;
};

void initializeOpenCL(cl::Device& device)
{
    std::vector<cl::Platform> platforms;
    std::vector<std::vector<cl::Device>> devices;
    cl::Platform::get(&platforms);
    
    devices.resize(platforms.size());
    std::size_t p = 0;
    for (const auto& platform : platforms)
    {
        platform.getDevices(CL_DEVICE_TYPE_ALL, &(devices[p]));
        p += 1;
    }

    if(platformIdx >= platforms.size()           ){ std::cout << "Invalid platform index: " << platformIdx << " of " << platforms.size() << "\n"; }
    if(deviceIdx   >= devices[platformIdx].size()){ std::cout << "Invalid device index: "   << deviceIdx   << " of " << devices[platformIdx].size() << "\n"; }
    
    std::cout << "Platform: " << platforms[platformIdx].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    
    device = devices[platformIdx][deviceIdx];
    std::cout << "Device: "   << device  .getInfo<CL_DEVICE_NAME>    () << std::endl;
}

cl::Program loadProgram(cl::Context ctx, std::string const& filename)
{
    // Load program source with manual preprocessing of #include due to nvidia bug...

    std::ifstream source_file{ filename };
    if ( !source_file.is_open() )
        throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + filename };

    std::ifstream include_file{ "particle.cl" };
    if ( !include_file.is_open() )
        throw std::runtime_error{ std::string{ "Cannot open kernel include: particle.cl" } };

    std::string src{ std::istreambuf_iterator<char>{ source_file },  std::istreambuf_iterator<char>{} };
    std::string inc{ std::istreambuf_iterator<char>{ include_file }, std::istreambuf_iterator<char>{} };

    std::string replace_what("#include <particle.cl>\n");
    std::size_t replace_where = src.find(replace_what);

    src.replace(replace_where, replace_what.size(), inc);

    // Create program and kernel
    return cl::Program{ ctx, src };
}

void generateParticles(std::vector<particle>& particles, size_t N)
{
    cl_float x_abs_range = 1.0f,
             y_abs_range = 1.0f,
             z_abs_range = 1.0f,
             mass_min = 1.0f,
             mass_max = 10.0f;

    // Create block of particles
    using uni = std::uniform_real_distribution<cl_float>;
    std::generate_n(
        std::back_inserter(particles),
        N,
        [prng = std::default_random_engine(42),
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
                0.0f },
            cl_float3{0.0f, 0.0f, 0.0f, 0.0f},
            m_dist(prng)
        };
    });
}

template<typename T>
T sq(T x){ return x*x; }

template<typename T>
T cube(T x){ return x*x*x; }

void cpuReference(std::vector<particle>const& particlesIn, std::vector<particle>& particlesOut)
{
    if(!runCPUReference){ return; }
    const auto N = particlesIn.size();
    particlesOut = particlesIn;

    for(std::size_t n = 0; n < nsteps; ++n)
    {
        for (auto IT = particlesOut.begin(); IT != particlesOut.end(); ++IT)
        {
            decltype(particle::f) force{ 0.0, 0.0, 0.0 };

            for (auto it = particlesOut.cbegin(); it != particlesOut.cend(); ++it)
            {
                if (IT != it)
                {
                    force.x += -G * IT->mass * it->mass * (IT->pos.x - it->pos.x) / cube(std::sqrt(sq(IT->pos.x - it->pos.x) + sq(IT->pos.y - it->pos.y) + sq(IT->pos.z - it->pos.z) ));
                    force.y += -G * IT->mass * it->mass * (IT->pos.y - it->pos.y) / cube(std::sqrt(sq(IT->pos.x - it->pos.x) + sq(IT->pos.y - it->pos.y) + sq(IT->pos.z - it->pos.z) ));
                    force.z += -G * IT->mass * it->mass * (IT->pos.z - it->pos.z) / cube(std::sqrt(sq(IT->pos.x - it->pos.x) + sq(IT->pos.y - it->pos.y) + sq(IT->pos.z - it->pos.z) ));
                }
            }

            IT->f = force;
        }

        for (auto& part : particlesOut)
        {
            const auto dt_per_m = deltat / part.mass;
            const auto dt_sq = sq(deltat);
            const auto half_dt_sq_per_m = 0.5f * dt_sq / part.mass;

            part.pos.x += part.v.x * deltat + part.f.x * half_dt_sq_per_m;
            part.pos.y += part.v.y * deltat + part.f.y * half_dt_sq_per_m;
            part.pos.z += part.v.z * deltat + part.f.z * half_dt_sq_per_m;

            part.v.x += part.f.x * dt_per_m;
            part.v.y += part.f.y * dt_per_m;
            part.v.z += part.f.z * dt_per_m;

            part.f = { 0.0, 0.0, 0.0 };
        }
    }
}

void compareResults(std::vector<particle>const& p1, std::vector<particle>const& p2)
{
    if(!runCPUReference){ return; }
    const float max_err = 5e-5f;
    auto floatComparator = [max_err](float l, float r){ return std::abs(l-r) < max_err; };
    auto particleComparator = [c=floatComparator](particle p1, particle p2)
    {
        return c(p1.pos.x, p2.pos.x) && c(p1.pos.y, p2.pos.y) && c(p1.pos.z, p2.pos.z) &&
               c(p1.v.x, p2.v.x)     && c(p1.v.y, p2.v.y)     && c(p1.v.z, p2.v.z);
    };

    for(int i=0; i<p1.size(); ++i)
	{
        if( !particleComparator(p1[i], p2[i]) )
        {
            std::cout << "[" << i << "] : "
            << p1[i].pos.x << "   " << p2[i].pos.x << "|"
            << p1[i].pos.y << "   " << p2[i].pos.y << "|"
            << p1[i].pos.z << "   " << p2[i].pos.z << "|"
            << p1[i].v.x   << "   " << p2[i].v.x << "|"
            << p1[i].v.y   << "   " << p2[i].v.y << "|"
            << p1[i].v.z   << "   " << p2[i].v.z
            << "\n";
        }
    }

    if(std::equal(p1.begin(), p1.end(), p2.begin(), particleComparator))
	{
		std::cout << "Success\n";
	}
	else{ std::cout << "Mismatch between CPU and GPU results.\n"; }
}