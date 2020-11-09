#include "common.hpp"

int main()
{
    cl::Program program;
    cl::Device device;
    try
    {
        cl::CommandQueue queue = cl::CommandQueue::getDefault();

        device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        std::cout << "Default queue on platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "Default queue on device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        program = loadProgram("./NBody-v2.cl");
        program.build({ device });

        auto interaction   = cl::Kernel(program, "interaction");
        auto forward_euler = cl::Kernel(program, "forward_euler");

        std::vector<particle> particles;
        generateParticles(particles, particle_count);

        cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, particles.size() * sizeof(particle), particles.data());

        interaction.setArg(0, buffer);
        forward_euler.setArg(0, buffer);
        forward_euler.setArg(1, 0.001f);

        // grid configuration:
        cl::NDRange interaction_gws = cl::NDRange(particles.size());
        cl::NDRange interaction_lws = cl::NullRange;
        cl::NDRange euler_gws = cl::NDRange(particles.size());
        cl::NDRange euler_lws = cl::NullRange;

        // Run warm-up kernels
        queue.enqueueNDRangeKernel(interaction,   cl::NullRange, interaction_gws, interaction_lws);
        queue.enqueueNDRangeKernel(forward_euler, cl::NullRange, euler_gws,       euler_lws);

        cl::finish();

        auto start = std::chrono::high_resolution_clock::now();

        for (std::size_t n = 0; n < 1000; ++n)
        {
            queue.enqueueNDRangeKernel(interaction,   cl::NullRange, interaction_gws, interaction_lws);
            queue.enqueueNDRangeKernel(forward_euler, cl::NullRange, euler_gws,       euler_lws);
        }

        cl::finish();

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds." << std::endl;
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

		if ( std::string(error.what()) == "clBuildProgram" )
		{
			if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) == CL_BUILD_ERROR)
				std::cerr << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>( device ) << std::endl;
		}

        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}