#include <clFFT-1D.hpp>

int main()
{
	// Test params
	std::size_t batch = 256;
	std::size_t N = 1024;

	// Host side container and init
	std::vector<std::complex<float>> x;
	std::vector<std::default_random_engine> prngs;
	std::uniform_real_distribution<float> dist;

	// OpenCL variables
	cl_int err = CL_SUCCESS;
	cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
	std::vector<cl::Platform> platforms;
	cl::Platform platform;
	std::vector<cl::Device> devices;
	std::array<cl_context_properties, 3> cprops;
	cl::Context context;
	std::vector<cl::CommandQueue> queues;
	std::vector<cl::Buffer> bufs_x;
	std::vector<Workflow> workflows;

	// clFFT variables
	std::vector<clfftPlanHandle> plans;
	clfftDim dim;
	std::array<std::size_t, 2> clLengths;
	clfftSetupData fftSetup;

	// OpenCL initialization
	std::cout << "Initializing OpenCL" << std::endl;
	err = cl::Platform::get(&platforms); checkerr(err, "cl::Platform::get");

	auto plat_it = std::find_if(platforms.cbegin(), platforms.cend(), [dev_type, &err](const cl::Platform& plat)
	{
		std::vector<cl::Device> devs;

		err = plat.getDevices(dev_type, &devs); checkerr(err, "cl::Platform::getDevices");
		return !devs.empty();
	});

	if (plat_it == platforms.cend())
	{
		std::cerr << "No platform found with desired device type. Exiting..." << std::endl;
		exit(EXIT_FAILURE);
	}

	platform = *plat_it;
	std::cout << "Selected platform:\n\n\t" << platform.getInfo<CL_PLATFORM_NAME>() << "\n" << std::endl;

	platform.getDevices(dev_type, &devices);
	std::cout << "Selected devices:\n\n";
	for (auto& device : devices)
		std::cout << "\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << std::endl;

	cprops = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()), 0 };

	context = cl::Context(devices, cprops.data(), nullptr, nullptr, &err); checkerr(err, "cl::Context::Context");

	queues.resize(devices.size());
	std::transform(devices.cbegin(), devices.cend(), queues.begin(), [&context](const cl::Device& dev) { return cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE/* | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/); });

	if (batch % devices.size() != 0)
	{
		std::cerr << "Device count is not a divisor of batch size. Exiting..." << std::endl;
		exit(EXIT_FAILURE);
	}

	bufs_x.resize(devices.size());
	for (auto& buf : bufs_x)
	{
		buf = cl::Buffer(context, CL_MEM_READ_WRITE, batch / devices.size() * N * N * sizeof(std::complex<float>), nullptr, &err); checkerr(err, "cl::Buffer::Buffer");
	}

	// Host-side initialization
	std::cout << "Generating random input" << std::endl;

	// Single threaded
	// std::generate_n(std::back_inserter(x), batch * N * N, [&prng, &dist]() { return dist(prng); });

	// Multi-threaded
	{
		std::vector<std::future<void>> futures;
		std::vector<std::vector<std::complex<float>>> temps(std::thread::hardware_concurrency());
		prngs.resize(std::thread::hardware_concurrency());

		for (unsigned int i = 0; i < std::thread::hardware_concurrency(); ++i)
			futures.push_back(std::async(std::launch::async, [=](std::reference_wrapper<std::vector<std::complex<float>>> temp, std::default_random_engine& prng, std::uniform_real_distribution<float> dist)
			{
				std::generate_n(std::back_inserter(temp.get()), batch * N * N / std::thread::hardware_concurrency(), [&prng, &dist]() { return dist(prng); });
			}, std::ref(temps.at(i)), prngs.at(i), dist));

		for (auto& future : futures) future.wait();
		for (auto& temp : temps) x.insert(x.end(), temp.begin(), temp.end());
	}

	// clFFT initialization
	std::cout << "Initializing clFFT" << std::endl;
	dim = CLFFT_2D;
	clLengths = { N, N };

	err = clfftInitSetupData(&fftSetup); checkerr(err, "clfftInitSetupData");
	err = clfftSetup(&fftSetup); checkerr(err, "clffftSetup");

	plans.resize(devices.size());
	for (std::size_t i = 0; i < plans.size(); ++i)
	{
		err = clfftCreateDefaultPlan(&plans.at(i), context(), dim, clLengths.data()); checkerr(err, "clCreateDefaultPlan");

		err = clfftSetPlanBatchSize(plans.at(i), batch / devices.size()); checkerr(err, "clfftSetPlanBatchSize");
		err = clfftSetPlanPrecision(plans.at(i), CLFFT_SINGLE); checkerr(err, "clfftSetPlanPrecision");
		err = clfftSetLayout(plans.at(i), CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED); checkerr(err, "clfftSetLayout");
		err = clfftSetResultLocation(plans.at(i), CLFFT_INPLACE); checkerr(err, "clfftSetResultLocation");

		// Bake plan
		err = cl::fft::bakePlan(plans.at(i), queues.at(i)); checkerr(err, "clfftBakePlan");
	}

	// Start time
	std::cout << "Starting " << batch << " count " << N << " * " << N << " std::complex<float> FFTs" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	{
		// Initiate data copy to devices
		workflows.resize(devices.size());
		for (std::size_t i = 0; i < devices.size(); ++i)
		{
			err = queues.at(i).enqueueWriteBuffer(bufs_x.at(i),
												  CL_FALSE,
												  0,
												  batch / devices.size() * N * N * sizeof(std::complex<float>),
												  x.data() + i * batch / devices.size() * N * N,
												  nullptr,
												  &workflows.at(i).events.at(Workflow::Events::Write));
			checkerr(err, "cl::CommandQueue::enqueueWriteBuffer");
			
			err = queues.at(i).enqueueMigrateMemObjects({ bufs_x.at(i) },
														0,
														nullptr,
														//&std::vector<cl::Event>{workflows.at(i).events.at(Workflow::Events::Write)},
														&workflows.at(i).events.at(Workflow::Events::Migrate));
			checkerr(err, "cl::CommandQueue::enqueueMigrateBuffer");
			
			err = queues.at(i).flush(); checkerr(err, "cl::CommandQueue::flush");
		}

		// Execute the plan
		for (std::size_t i = 0; i < plans.size(); ++i)
		{
			err = cl::fft::enqueueTransform(plans.at(i),
											CLFFT_FORWARD,
											queues.at(i),
											{},
											workflows.at(i).events.at(Workflow::Events::Exec),
											bufs_x.at(i));
		}

		// Initiate data fetch from devices
		for (std::size_t i = 0; i < devices.size(); ++i)
		{
			err = queues.at(i).enqueueReadBuffer(bufs_x.at(i),
												 CL_FALSE,
												 0,
												 batch / devices.size() * N * N * sizeof(std::complex<float>),
												 x.data() + i * batch / devices.size() * N * N,
												 nullptr,
												 &workflows.at(i).events.at(Workflow::Events::Read));
			checkerr(err, "cl::CommandQueue::enqueueReadBuffer");

			err = queues.at(i).flush(); checkerr(err, "cl::CommandQueue::flush");
		}

		// Wait for copy to complete
		for (auto& queue : queues)
		{
			err = queue.finish(); checkerr(err, "cl::CommandQueue::finish");
		}
	}
	// End time
	auto end = std::chrono::high_resolution_clock::now();

	// Display timings
	std::cout << "Total time as measured by std::chrono::high_precision_timer =\n\n\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds.\n" << std::endl;

	//report_workflow_stage<Workflow::Events::Write,   CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_END>("Host-device init as measured by cl::Event::getProfilingInfo", workflows);
	report_workflow_stage<Workflow::Events::Migrate, CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END>("Host-device copy as measured by cl::Event::getProfilingInfo", workflows);
	report_workflow_stage<Workflow::Events::Exec,    CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END>("Fourier transform as measured by cl::Event::getProfilingInfo", workflows);
	report_workflow_stage<Workflow::Events::Read,    CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_END>("Device-host copy as measured by cl::Event::getProfilingInfo", workflows);

	// Release non-RAII resources in reverse order
	for (auto& plan : plans) err = clfftDestroyPlan(&plan);
	clfftTeardown();

	return 0;
}