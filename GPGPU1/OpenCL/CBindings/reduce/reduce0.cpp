#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

const std::string kenrel_filename = "reduce0.cl";

int main()
{
    const size_t sz = 256*256;
    const size_t block_sz = 256;
    const int    n_blocks = sz / block_sz;

	std::vector<float> A(sz);
	std::vector<float> B(n_blocks);

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};

    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(A.begin(), A.end(), gen);
	
    cl_int status = CL_SUCCESS;

    cl_uint numPlatforms = 0;
    std::vector<cl_platform_id> platforms;
    std::vector<std::vector<cl_device_id>> devices;
    
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if(status != CL_SUCCESS){ std::cout << "Cannot get number of platforms: " << status << "\n"; return -1; }
    
    platforms.resize(numPlatforms);
    devices.resize(numPlatforms);
	status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if(status != CL_SUCCESS){ std::cout << "Cannot get platform ids: " << status << "\n"; return -1; }

    for(cl_uint i=0; i<numPlatforms; ++i)
    {
        cl_uint numDevices = 0;
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        if(status != CL_SUCCESS){ std::cout << "Cannot get number of devices: " << status << "\n"; return -1; }

        if(numDevices == 0){ std::cout << "There are no devices in platform " << i << "\n"; continue; }

        devices[i].resize(numDevices);
        
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices[i].data(), nullptr);
        if(status != CL_SUCCESS){ std::cout << "Cannot get device ids: " << status << "\n"; return -1; }
    }

    //select platform and device:
    const auto platformIdx = 1;
    const auto deviceIdx   = 0;
    const auto platform    = platforms[platformIdx];
    const auto device      = devices[platformIdx][deviceIdx];

    //print names:
    {
        size_t vendor_name_length = 0;
        status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendor_name_length);
        if(status != CL_SUCCESS){ std::cout << "Cannot get platform vendor name length: " << status << "\n"; return -1; }

        std::string vendor_name(vendor_name_length, '\0');
        status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendor_name_length, (void*)vendor_name.data(), nullptr);
        if(status != CL_SUCCESS){ std::cout << "Cannot get platform vendor name: " << status << "\n"; return -1; }

        size_t device_name_length = 0;
        status = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_length);
        if(status != CL_SUCCESS){ std::cout << "Cannot get device name length: " << status << "\n"; return -1; }

        std::string device_name(device_name_length, '\0');
        status = clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_length, (void*)device_name.data(), nullptr);
        if(status != CL_SUCCESS){ std::cout << "Cannot get device name: " << status << "\n"; return -1; }

        std::cout << "Platform: " << vendor_name << "\n";
        std::cout << "Device: "   << device_name << "\n";
    }

	std::array<cl_context_properties, 3> cps = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	auto context = clCreateContext(cps.data(), 1, &device, 0, 0, &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create context: " << status << "\n"; return -1; }

    //OpenCL 1.2:
    auto queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    //Above OpenCL 1.2:
    //cl_command_queue_properties cqps = CL_QUEUE_PROFILING_ENABLE;
	//std::array<cl_queue_properties, 3> qps = { CL_QUEUE_PROPERTIES, cqps, 0 };
	//auto queue = clCreateCommandQueueWithProperties(context, device, qps.data(), &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create command queue: " << status << "\n"; return -1; }

	std::ifstream file(kenrel_filename.data());
	std::string source( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	size_t      sourceSize = source.size();
	const char* sourcePtr  = source.c_str();
	auto program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create program: " << status << "\n"; return -1; }

	status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
	if (status != CL_SUCCESS)
	{
        std::cout << "Cannot build program: " << status << "\n";
		size_t len = 0;
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
		std::unique_ptr<char[]> log = std::make_unique<char[]>(len);
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.get(), nullptr);
		std::cout << log.get() << "\n";
		return -1;
	}

	auto kernel = clCreateKernel(program, "reduce0", &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create kernel: " << status << "\n"; return -1; }

    float  gpu_sum = 0.0f;

    auto bufferSrc = clCreateBuffer(context, CL_MEM_READ_ONLY   | CL_MEM_COPY_HOST_PTR, sz*sizeof(float),         A.data(), &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create buffer src: " << status << "\n"; return -1; }
	auto bufferDst = clCreateBuffer(context, CL_MEM_READ_WRITE  | CL_MEM_HOST_NO_ACCESS,  n_blocks*sizeof(float), nullptr,  &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create buffer dst: " << status << "\n"; return -1; }
	auto bufferRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY  | CL_MEM_HOST_READ_ONLY,  sizeof(float),          nullptr,  &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create buffer res: " << status << "\n"; return -1; }

    cl_event evt[2];

    {
        size_t gsize          = n_blocks * block_sz;
	    size_t lsize          = block_sz;
        size_t local_mem_size = lsize * sizeof(float);
        cl_int gsize_as_int   = (cl_int)gsize;
        
        status = clSetKernelArg(kernel, 0, sizeof(bufferDst), &bufferDst); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 0: " << status << "\n"; return -1; }
	    status = clSetKernelArg(kernel, 1, sizeof(bufferSrc), &bufferSrc); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 1: " << status << "\n"; return -1; }
	    status = clSetKernelArg(kernel, 2, local_mem_size, nullptr);       if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 2: " << status << "\n"; return -1; }
	    status = clSetKernelArg(kernel, 3, sizeof(cl_int), &gsize_as_int); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 3: " << status << "\n"; return -1; }

        status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gsize, &lsize, 0, nullptr, &evt[0]);
        if(status != CL_SUCCESS){ std::cout << "Cannot enqueue kernel (1): " << status << "\n"; return -1; }
    }
	
    {
        size_t gsize          = block_sz;
	    size_t lsize          = block_sz;
        size_t local_mem_size = lsize * sizeof(float);
        cl_int gsize_as_int   = (cl_int)gsize;

        status = clSetKernelArg(kernel, 0, sizeof(bufferRes), &bufferRes); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 0: " << status << "\n"; return -1; }
	    status = clSetKernelArg(kernel, 1, sizeof(bufferDst), &bufferDst); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 1: " << status << "\n"; return -1; }
	    status = clSetKernelArg(kernel, 2, local_mem_size, nullptr);       if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 2: " << status << "\n"; return -1; }
	    status = clSetKernelArg(kernel, 3, sizeof(cl_int), &gsize_as_int); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel arg 3: " << status << "\n"; return -1; }

        status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gsize, &lsize, 1, &evt[0], &evt[1]);
        if(status != CL_SUCCESS){ std::cout << "Cannot enqueue kernel (2): " << status << "\n"; return -1; }
    }

    status = clEnqueueReadBuffer(queue, bufferRes, CL_TRUE, 0, 1 * sizeof(float), &gpu_sum, 1, &evt[1], nullptr);
    if(status != CL_SUCCESS){ std::cout << "Cannot read buffer: " << status << "\n"; return -1; }
	
    float dt1 = 0.0f, dt2 = 0.0f;//milliseconds
    {
        cl_ulong t1_0, t1_1, t2_0, t2_1;
        status = clGetEventProfilingInfo(evt[0], CL_PROFILING_COMMAND_START, sizeof(t1_0), &t1_0, nullptr);
        status = clGetEventProfilingInfo(evt[0], CL_PROFILING_COMMAND_END, sizeof(t1_1), &t1_1, nullptr);
        dt1 = (t1_1 - t1_0)*0.001f*0.001f;

        status = clGetEventProfilingInfo(evt[1], CL_PROFILING_COMMAND_START, sizeof(t2_0), &t2_0, nullptr);
        status = clGetEventProfilingInfo(evt[1], CL_PROFILING_COMMAND_END, sizeof(t2_1), &t2_1, nullptr);
        dt2 = (t2_1 - t2_0)*0.001f*0.001f;
    }

    clReleaseEvent(evt[0]);
    clReleaseEvent(evt[1]);
    clReleaseMemObject(bufferSrc);
    clReleaseMemObject(bufferDst);
    clReleaseMemObject(bufferRes);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);

    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_sum = std::accumulate(A.begin(), A.end(), 0.0f);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout.precision(10);
    std::cout << "cpu_sum = " << cpu_sum << "\n";
    std::cout << "gpu_sum = " << gpu_sum << "\n";

    float rel_err = std::abs((cpu_sum - gpu_sum) / cpu_sum);
    std::cout << "Relative error is: " << rel_err << "\n";
	if( rel_err < 2e-4 )
	{
		std::cout << "Success.\n";
	}
	else
	{
		std::cout << "Mismatch in CPU and GPU result.\n";
    }
    
    std::cout << "CPU Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "GPU Computation took: " << dt1 << " + " << dt2 << " = " << dt1+dt2 << " ms.\n";
	return 0;
}
