#include <OpenCL-MatrixMul-C-API.hpp>

int main()
{
	cl_platform_id platform = NULL;
	auto status = clGetPlatformIDs(1, &platform, NULL);

    cl_device_id devices[2] = {NULL, NULL};
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, devices, NULL);

    auto device = devices[1];

	cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	auto context = clCreateContext(cps, 1, &device, 0, 0, &status);

	cl_command_queue_properties cqps = CL_QUEUE_PROFILING_ENABLE;
	cl_queue_properties qps[] = { CL_QUEUE_PROPERTIES, cqps, 0 };
	auto queue = clCreateCommandQueueWithProperties(context, device, &qps[0], &status);

	std::ifstream file(kernel_location.c_str());
	std::string source( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	size_t      sourceSize = source.size();
	const char* sourcePtr  = source.c_str();
	auto program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &status);

	status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
	if (status != CL_SUCCESS)
	{
		size_t len = 0;
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
		std::unique_ptr<char[]> log = std::make_unique<char[]>(len);
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.get(), nullptr);
		std::cout << log.get() << "\n";
		return -1;
	}

	auto kernel1 = clCreateKernel(program, "matmul0", &status);
	auto kernel2 = clCreateKernel(program, "matmul1", &status);

	static const int size = 1024;
	static const int blocksize = 8;
	std::vector<double> A(size*size), B(size*size), C(size*size), D(size*size), E(size*size);

	std::random_device rnd_device;
	std::mt19937 mersenne_engine(rnd_device());
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	auto gen = [&]() { return dist(mersenne_engine); };
	std::generate(A.begin(), A.end(), gen );
	std::generate(B.begin(), B.end(), gen );
	std::fill(C.begin(), C.end(), 0.0f);
	std::fill(D.begin(), D.end(), 0.0f);
    std::fill(E.begin(), E.end(), 0.0f);

    //naive implementation:
    auto tp1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size; ++j)
        {
            auto acc = 0.0;
            for(int k=0; k<size; ++k)
            {
                acc += A[i*size+k] * B[k*size+j];
            }
            E[i*size+j] = acc;
        }
    }
    auto tp2 = std::chrono::high_resolution_clock::now();
    auto tnaive = std::chrono::duration_cast<std::chrono::microseconds>(tp2-tp1).count()/1000.0;
	
	auto bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(double), A.data(), &status);
	auto bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(double), B.data(), &status);
	auto bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, C.size() * sizeof(double), C.data(), &status);
	auto bufferD = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, D.size() * sizeof(double), D.data(), &status);

	status = clSetKernelArg(kernel1, 0, sizeof(bufferA), &bufferA);
	status = clSetKernelArg(kernel1, 1, sizeof(bufferB), &bufferB);
	status = clSetKernelArg(kernel1, 2, sizeof(bufferC), &bufferC);
	status = clSetKernelArg(kernel1, 3, sizeof(int), &size);

	status = clSetKernelArg(kernel2, 0, sizeof(bufferA), &bufferA);
	status = clSetKernelArg(kernel2, 1, sizeof(bufferB), &bufferB);
	status = clSetKernelArg(kernel2, 2, sizeof(bufferD), &bufferD);
	status = clSetKernelArg(kernel2, 3, sizeof(int), &size);
	status = clSetKernelArg(kernel2, 4, sizeof(int), &blocksize);
	status = clSetKernelArg(kernel2, 5, blocksize*blocksize*sizeof(double), nullptr);
	status = clSetKernelArg(kernel2, 6, blocksize*blocksize*sizeof(double), nullptr);

	size_t gsizes[] = { size, size, 0 };
	size_t lsizes[] = { blocksize, blocksize, 0 };

	cl_event ev1, ev2;
	cl_ulong t1_0, t1_1, t2_0, t2_1;

	status = clEnqueueNDRangeKernel( queue, kernel1, 2, nullptr, gsizes, nullptr, 0, nullptr, &ev1);
	status = clWaitForEvents(1, &ev1);
	status = clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_START, sizeof(t1_0), &t1_0, nullptr);
	status = clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_END, sizeof(t1_1), &t1_1, nullptr);

	status = clEnqueueReadBuffer(queue, bufferC, false, 0, C.size() * sizeof(double), C.data(), 0, nullptr, nullptr);

	status = clEnqueueNDRangeKernel( queue, kernel2, 2, nullptr, gsizes, lsizes, 0, nullptr, &ev2);
	status = clWaitForEvents(1, &ev2);
	status = clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_START, sizeof(t2_0), &t2_0, nullptr);
	status = clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_END, sizeof(t2_1), &t2_1, nullptr);
	status = clEnqueueReadBuffer(queue, bufferD, false, 0, D.size() * sizeof(double), D.data(), 0, nullptr, nullptr);
	status = clFinish(queue);
	
    std::cout << "Naive    took " << tnaive << " msecs\n";
	std::cout << "Kernel 0 took " << (t1_1 - t1_0)*0.001*0.001 << " msecs\n";
	std::cout << "Kernel 1 took " << (t2_1 - t2_0)*0.001*0.001 << " msecs\n";

    auto checker = [&](std::vector<double> const& u, std::vector<double> const& v)
    {
        return std::inner_product(u.cbegin(), u.cend(), v.cbegin(), true, [](bool const& b, auto const& x){ return b && x < 1e-10; }, [](auto const& ref, auto const& x){ return std::abs(ref-x); } );
    };
    auto res1 = checker(E, C);
    auto res2 = checker(E, D);
    auto res3 = checker(C, D);

	std::cout << "Result1: " << std::string(res1 ? "PASSED" : "FAILED") << "\n";
    std::cout << "Result2: " << std::string(res2 ? "PASSED" : "FAILED") << "\n";
    std::cout << "Result3: " << std::string(res3 ? "PASSED" : "FAILED") << "\n";

    clReleaseEvent(ev1);
    clReleaseEvent(ev2);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseMemObject(bufferD);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);

	return 0;
}