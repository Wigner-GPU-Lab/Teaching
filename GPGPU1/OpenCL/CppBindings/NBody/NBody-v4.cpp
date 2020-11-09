#include "common.hpp"

int main()
{
    cl::Device device;
    cl::Program program;
    try
    {
        initializeOpenCL(device);

        cl::Context context = cl::Context(device);
        cl::CommandQueue queue = cl::CommandQueue(context, device);

        program = loadProgram(context, "./NBody-v4.cl");
        program.build({ device });

        auto interaction   = cl::Kernel(program, "interaction");
        auto forward_euler = cl::Kernel(program, "forward_euler");

        std::vector<particle> particles, particlesInit;
        generateParticles(particles, particle_count);
        particlesInit = particles;

        cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, particles.size() * sizeof(particle), particles.data());

        size_t local_count = 256;

        interaction.setArg(0, buffer);
        interaction.setArg(1, cl::Local(local_count * sizeof(particle)) );
        forward_euler.setArg(0, buffer);
        forward_euler.setArg(1, deltat);

        // grid configuration:
        cl::NDRange interaction_gws = cl::NDRange(particles.size());
        cl::NDRange interaction_lws = cl::NDRange(local_count);
        cl::NDRange euler_gws = cl::NDRange(particles.size());
        cl::NDRange euler_lws = cl::NullRange;

        // Run warm-up kernels
        queue.enqueueNDRangeKernel(interaction,   cl::NullRange, interaction_gws, interaction_lws);
        queue.enqueueNDRangeKernel(forward_euler, cl::NullRange, euler_gws,       euler_lws);
        cl::copy(queue, std::begin(particles), std::end(particles), buffer);
        cl::finish();

        auto start = std::chrono::high_resolution_clock::now();

        for (std::size_t n = 0; n < nsteps; ++n)
        {
            queue.enqueueNDRangeKernel(interaction,   cl::NullRange, interaction_gws, interaction_lws);
            queue.enqueueNDRangeKernel(forward_euler, cl::NullRange, euler_gws,       euler_lws);
        }
        cl::copy(queue, buffer, std::begin(particles), std::end(particles));

        cl::finish();

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds." << std::endl;

        std::vector<particle> particlesRef;
        cpuReference(particlesInit, particlesRef);
        compareResults(particlesRef, particles);
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