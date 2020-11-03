// STL includes
#include <chrono>
#include <numeric>
#include <iostream>
#include <sstream>

// TCLAP includes
#include <tclap/CmdLine.h>

// Manybody includes
#include <particle.hpp>

int main(int argc, char** argv)
{
	std::string banner = "Manybody OpenCL v1: N^2, cache unaware, aliased, recalculating";
	TCLAP::CmdLine cli( banner );

    TCLAP::ValueArg<std::string> input_arg("i", "input", "Path to input file", true, "./", "path");
    TCLAP::ValueArg<std::string> output_arg("o", "output", "Path to output file", false, "", "path");
    TCLAP::ValueArg<std::string> validate_arg("v", "validate", "Path to validation file", false, "", "path");
	TCLAP::ValueArg<std::string> type_arg( "t", "type", "Device type to use", false, "cpu", "cpu|gpu|acc" );
	TCLAP::ValueArg<std::size_t> device_arg( "d", "device", "Device id to use", false, 0, "non-negative integral" );
	TCLAP::ValueArg<std::size_t> platform_arg( "p", "platform", "Platform id to use", false, 0, "non-negative integral" );
    TCLAP::ValueArg<std::size_t> iterate_arg("n", "", "Number of iterations to take", false, 1, "positive integral");
    TCLAP::SwitchArg quiet_arg("q", "quiet", "Suppress standard output", false);

	cli.add( input_arg );
	cli.add( output_arg );
	cli.add( validate_arg );
	cli.add( type_arg );
	cli.add( device_arg );
	cli.add( platform_arg );
	cli.add( iterate_arg );
	cli.add( quiet_arg );

    try
    {
        cli.parse(argc, argv);
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue command_queue;
	cl::Program program;
	cl::Buffer buffer;
    cl::Kernel interaction, forward_euler;
    cl::NDRange interaction_gws, interaction_lws, euler_gws, euler_lws;
    cl::Event interaction_event, euler_event;

	std::vector<particle> particles;

	if ( !quiet_arg.getValue() ) std::cout << banner << std::endl;

	try
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get( &platforms );

		if ( platforms.empty() )
		{
			std::cerr << "No OpenCL platform detected." << std::endl;
			exit( EXIT_FAILURE );
		}

		platform = platforms.at( platform_arg.getValue() );

		if ( !quiet_arg.getValue() ) std::cout << "Selected platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

		cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
		if ( type_arg.getValue() == "cpu" ) device_type = CL_DEVICE_TYPE_CPU;
		if ( type_arg.getValue() == "gpu" ) device_type = CL_DEVICE_TYPE_GPU;
		if ( type_arg.getValue() == "acc" ) device_type = CL_DEVICE_TYPE_ACCELERATOR;

		std::vector<cl::Device> devices;
		platform.getDevices( device_type, &devices );

		device = devices.at( device_arg.getValue() );

		if ( !quiet_arg.getValue() ) std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

		if ( (device.getInfo<CL_DEVICE_EXTENSIONS>().find( "cl_khr_fp64" ) == std::string::npos) &&
			(device.getInfo<CL_DEVICE_EXTENSIONS>().find( "cl_amd_fp64" ) == std::string::npos) )
		{
			std::cerr << "Selected device does not support double precision" << std::endl;
			exit( EXIT_FAILURE );
		}

		std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()), 0 };
		context = cl::Context( device, props.data(), nullptr, nullptr);

		command_queue = cl::CommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE);

        std::ifstream source_file(std::string(MANYBODY_OPENCL_KERNEL_PATH) + "manybody-cl-v1-single-device.cl");
		if ( !source_file.is_open() )
			throw cl::Error(-9999, "std::ifstream::is_open");

		std::string source_string( std::istreambuf_iterator<char>( source_file ), (std::istreambuf_iterator<char>()) );

        // Workaround for buggy -I switch on Nvidia platform
        std::string replace_what("<particle.cl>");
        std::string replace_with(std::string("<") + std::string(MANYBODY_OPENCL_KERNEL_PATH) + "../inc/particle.cl>");
        std::size_t replace_where = source_string.find(replace_what);

        source_string.replace(replace_where,
                              replace_what.size(),
                              replace_with);

		program = cl::Program( context, source_string );

        std::stringstream build_opts;

        build_opts <<
            "-cl-mad-enable " <<
            "-cl-no-signed-zeros " <<
            "-cl-finite-math-only " <<
            "-cl-single-precision-constant " <<
            "-I " << std::string(MANYBODY_OPENCL_KERNEL_PATH) + "../inc";

		if ( !quiet_arg.getValue() ){ std::cout << "Building program..."; std::cout.flush(); }
		program.build( std::vector<cl::Device>( 1, device ), build_opts.str().c_str());
		if ( !quiet_arg.getValue() ){ std::cout << " done." << std::endl; }

        interaction = cl::Kernel(program, "interaction");
        forward_euler = cl::Kernel(program, "forward_euler");

        if (!quiet_arg.getValue())
            std::cout << "Interaction kernel preferred WGS: " << interaction.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << std::endl;
        if (!quiet_arg.getValue())
            std::cout << "Forward Euler kernel preferred WGS: " << forward_euler.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << std::endl;

        if (!quiet_arg.getValue()){ std::cout << "Reading input file..."; std::cout.flush(); }
		particles = read_particle_file( input_arg.getValue() );
        if (!quiet_arg.getValue()){ std::cout << " done." << std::endl; }

        buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, particles.size() * sizeof(particle), particles.data());

        interaction.setArg(0, buffer);
        forward_euler.setArg(0, buffer);
        forward_euler.setArg(1, 0.001f);

        interaction_gws = cl::NDRange(particles.size());
        interaction_lws = cl::NullRange;
        euler_gws = cl::NDRange(particles.size());
        euler_lws = cl::NullRange;

        // Run warm-up kernels
        command_queue.enqueueNDRangeKernel(interaction, cl::NullRange, interaction_gws, interaction_lws, nullptr, &interaction_event);
        command_queue.enqueueNDRangeKernel(forward_euler, cl::NullRange, euler_gws, euler_lws);

        command_queue.finish();

        // Reset data
        command_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, particles.size() * sizeof(particle), particles.data());
	}
	catch ( cl::Error error )
	{
		std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

		if ( std::string(error.what()) == "clBuildProgram" )
		{
			if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) == CL_BUILD_ERROR)
				std::cerr << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>( device ) << std::endl;
		}

		exit( error.err() );
	}
    
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (std::size_t n = 0; n < iterate_arg.getValue(); ++n)
        {
            try
            {
                command_queue.enqueueNDRangeKernel(interaction, cl::NullRange, interaction_gws, interaction_lws, nullptr, &interaction_event);
                command_queue.enqueueNDRangeKernel(forward_euler, cl::NullRange, euler_gws, euler_lws);
            }
            catch (cl::Error error)
            {
                std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
                exit(error.err());
            }
        }

        command_queue.finish();

        auto end = std::chrono::high_resolution_clock::now();
        if (!quiet_arg.getValue()) std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;
    }

    {
        if (output_arg.getValue() != "")
        {
            if (!quiet_arg.getValue()) std::cout << "Exporting results into " << output_arg.getValue() << std::endl;

            command_queue.enqueueReadBuffer(buffer, true, 0, particles.size() * sizeof(particle), particles.data());

            write_validation_file(particles.cbegin(), particles.cend(), output_arg.getValue());
        }
    }

    {
        if (validate_arg.getValue() != "")
        {
            if (!quiet_arg.getValue()) std::cout << "Validating results against " << validate_arg.getValue() << std::endl;

            command_queue.enqueueReadBuffer(buffer, true, 0, particles.size() * sizeof(particle), particles.data());

            if (!validate(particles.cbegin(), particles.cend(), validate_arg.getValue())) exit(EXIT_FAILURE);
        }
    }

	return 0;
}