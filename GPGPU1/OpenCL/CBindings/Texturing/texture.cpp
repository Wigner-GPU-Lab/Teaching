#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

struct rawcolor{ unsigned char r, g, b, a; };
struct rawcolor3{ unsigned char r, g, b; };
struct color   { float         r, g, b, a; };

int main()
{
    static const std::string input_filename   = "input.png";

    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components

    // Load image:
    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }

    std::vector<color> input(w*h);
    std::vector<color> output(w*h);
    if(ch == 4)
    {
        std::transform(data0, data0+w*h, input.begin(), [](rawcolor c){ return color{c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f}; } );
    }
    else if(ch == 3)
    {
        std::transform(data0, data0+w*h, input.begin(), [](rawcolor c){ return color{c.r/255.0f, c.g/255.0f, c.b/255.0f, 1.0f}; } );
    }
    stbi_image_free(data0);

    // OpenCL init:

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

	std::ifstream file("sobel.cl");
	if(!file.is_open()){ std::cout << "Could not open kernel file: sobel.cl\n"; return -1; };
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

	auto kernel = clCreateKernel(program, "sobel", &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create kernel: " << status << "\n"; return -1; }
	
    cl_image_format format = { CL_RGBA, CL_FLOAT };
	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width =  w; //x
	desc.image_height = h; //y
	desc.image_depth =  0;
	
	cl_mem img_src = clCreateImage(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, &format, &desc, input.data(), &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create source image object: " << status << "\n"; return -1; }
    cl_mem img_dst = clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,                          &format, &desc, input.data(), &status);
	if(status != CL_SUCCESS){ std::cout << "Cannot create destination image object: " << status << "\n"; return -1; }

	status = clSetKernelArg(kernel, 0, sizeof(img_src), &img_src);
    if(status != CL_SUCCESS){ std::cout << "Cannot set kernel argument 0: " << status << "\n"; return -1; }
	status = clSetKernelArg(kernel, 1, sizeof(img_dst), &img_dst);
    if(status != CL_SUCCESS){ std::cout << "Cannot set kernel argument 1: " << status << "\n"; return -1; }

    size_t kernel_dims[2] = {(size_t)w, (size_t)h};
	status = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, kernel_dims, nullptr, 0, nullptr, nullptr);
    if(status != CL_SUCCESS){ std::cout << "Cannot enqueue kernel: " << status << "\n"; return -1; }
    
    size_t origin[3] = {0, 0, 0};
    size_t dims[3] = {(size_t)w, (size_t)h, 1};
	status = clEnqueueReadImage(queue, img_dst, false, origin, dims, 0, 0, output.data(), 0, nullptr, nullptr);
	if(status != CL_SUCCESS){ std::cout << "Cannot read back image: " << status << "\n"; return -1; }

    status = clFinish(queue);
    if(status != CL_SUCCESS){ std::cout << "Cannot finish: " << status << "\n"; return -1; }
    {
        std::vector<rawcolor> tmp(w*h*4);
        std::transform(output.cbegin(), output.cend(), tmp.begin(),
                [](color c){ return rawcolor{   (unsigned char)(c.r*255.0f),
                                                (unsigned char)(c.g*255.0f),
                                                (unsigned char)(c.b*255.0f),
                                                (unsigned char)(1.0f*255.0f) }; } );

            int res = stbi_write_png("result.png", w, h, 4, tmp.data(), w*4);
            if(res == 0){ std::cout << "Error writing output to file\n"; }
            else        { std::cout << "Output written to file\n"; }
    }

    clReleaseMemObject(img_src);
    clReleaseMemObject(img_dst);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
	
	return 0;
}
