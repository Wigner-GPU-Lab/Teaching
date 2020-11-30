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
#include <iostream>
#include <fstream>

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

struct color{ unsigned char r, g, b, a; };
struct three_histograms
{
    std::array<unsigned int, 256> rh, gh, bh;
    void make_null()
    {
        for(int i=0; i<256; ++i)
        {
            rh[i] = 0; gh[i] = 0; bh[i] = 0;
        }
    }

    void fromLinearMemory( std::vector<unsigned int>& input )
    {
        for(int i=0; i<256; ++i)
        {
            rh[i] = input[0*256+i];
            gh[i] = input[1*256+i];
            bh[i] = input[2*256+i];
        }
    }
};

void cpu_histo( three_histograms& output, color* const& input, int W, int H )
{
    for(int y=0; y<H; ++y)
    {
        for(int x=0; x<W; ++x)
        {
            color c = input[y*W+x];
            output.rh[c.r] += 1;
            output.gh[c.g] += 1;
            output.bh[c.b] += 1;
        }
    }
}

int main()
{
    static const std::string input_filename   = "NZ.jpg";
    static const std::string output_filename1 = "cpu_out.jpg";
    static const std::string output_filename2 = "gpu_out1.jpg";
    static const std::string output_filename3 = "gpu_out2.jpg";

    static const int block_size = 16;
    //int nBlocksW = 0; //number of blocks horizontally, not used now
    int nBlocksH = 0; //number of blocks vertically
    
    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components

    color* data0 = reinterpret_cast<color*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        //nBlocksW = w / block_size; //not used now
        nBlocksH = h / block_size;
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }
    
    three_histograms cpu;  cpu.make_null();
    three_histograms gpu1; gpu1.make_null();
    three_histograms gpu2; gpu2.make_null();
   
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_histo(cpu, data0, w, h);
    auto t1 = std::chrono::high_resolution_clock::now();

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
    const auto platformIdx = 0;
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

	std::ifstream file("atomics.cl");
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

	auto kernel_global_atomics = clCreateKernel(program, "gpu_histo_global_atomics", &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create kernel 'gpu_histo_global_atomics': " << status << "\n"; return -1; }

    auto kernel_accumulate = clCreateKernel(program, "gpu_histo_accumulate", &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create kernel 'gpu_histo_accumulate': " << status << "\n"; return -1; }

    auto kernel_shared_atomics = clCreateKernel(program, "gpu_histo_shared_atomics", &status);
    if(status != CL_SUCCESS){ std::cout << "Cannot create kernel 'gpu_histo_shared_atomics': " << status << "\n"; return -1; }

    //GPU version using global atomics:
    float dt1 = 0.0f;
    {
        auto bufferInput    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w*h*sizeof(color), data0, &status);
        if(status != CL_SUCCESS){ std::cout << "Cannot create input buffer: " << status << "\n"; return -1; }
        
        auto bufferPartials = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,  nBlocksH*3*256*sizeof(unsigned int), nullptr,  &status);
        if(status != CL_SUCCESS){ std::cout << "Cannot create partial buffer: " << status << "\n"; return -1; }
        
        cl_uchar4 zero = {0, 0, 0, 0};
        status = clEnqueueFillBuffer(queue, bufferPartials, &zero, sizeof(cl_uchar4), 0, nBlocksH*3*256*sizeof(unsigned int), 0, nullptr, nullptr);
        if(status != CL_SUCCESS){ std::cout << "Cannot zero partial buffer: " << status << "\n"; return -1; }

        status = clFinish(queue);
        if(status != CL_SUCCESS){ std::cout << "Cannot finish queue: " << status << "\n"; return -1; }

        auto bufferOutput  = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,  3*256*sizeof(unsigned int), nullptr,  &status);
        if(status != CL_SUCCESS){ std::cout << "Cannot create output buffer: " << status << "\n"; return -1; }

        cl_event evt[2];

        //First kernel of global histograms:
        {
            status = clSetKernelArg(kernel_global_atomics, 0, sizeof(bufferPartials), &bufferPartials); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 1 arg 0: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_global_atomics, 1, sizeof(bufferInput),    &bufferInput);    if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 1 arg 1: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_global_atomics, 2, sizeof(int),            &w);              if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 1 arg 2: " << status << "\n"; return -1; }

            size_t kernel_global_size[2] = {(size_t)block_size, (size_t)nBlocksH*block_size};
            size_t kernel_local_size[2] = {(size_t)block_size, (size_t)block_size};
	        status = clEnqueueNDRangeKernel(queue, kernel_global_atomics, 2, nullptr, kernel_global_size, kernel_local_size, 0, nullptr, &evt[0]);
            if(status != CL_SUCCESS){ std::cout << "Cannot enqueue kernel 1: " << status << "\n"; return -1; }
        }

        //Second kernel: accumulate partial results:
        {
            status = clSetKernelArg(kernel_accumulate, 0, sizeof(bufferOutput),   &bufferOutput);   if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 2 arg 0: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_accumulate, 1, sizeof(bufferPartials), &bufferPartials); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 2 arg 1: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_accumulate, 2, sizeof(int),            &nBlocksH);       if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 2 arg 2: " << status << "\n"; return -1; }

            size_t kernel_global_size[1] = {(size_t)(3*256)};
	        status = clEnqueueNDRangeKernel(queue, kernel_accumulate, 1, nullptr, kernel_global_size, nullptr, 0, nullptr, &evt[1]);
            if(status != CL_SUCCESS){ std::cout << "Cannot enqueue kernel 2: " << status << "\n"; return -1; }
        }

        std::vector<unsigned int> tmp(3*256);
        
        status = clEnqueueReadBuffer(queue, bufferOutput, CL_TRUE, 0, 3*256*sizeof(unsigned int), tmp.data(), 1, &evt[1], nullptr);
        if(status != CL_SUCCESS){ std::cout << "Cannot read buffer: " << status << "\n"; return -1; }

        cl_ulong t1_0, t1_1;
        status = clGetEventProfilingInfo(evt[0], CL_PROFILING_COMMAND_START, sizeof(t1_0), &t1_0, nullptr);
        status = clGetEventProfilingInfo(evt[1], CL_PROFILING_COMMAND_END,   sizeof(t1_1), &t1_1, nullptr);
        dt1 = (t1_1 - t1_0)*0.001f*0.001f;

        clReleaseEvent(evt[0]);
        clReleaseEvent(evt[1]);

        gpu1.fromLinearMemory(tmp);

        clReleaseMemObject(bufferInput);
        clReleaseMemObject(bufferPartials);
        clReleaseMemObject(bufferOutput);
    }

    //GPU version using shared atomics:
    float dt2 = 0.0f;
    {
        auto bufferInput    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w*h*sizeof(color), data0, &status);
        if(status != CL_SUCCESS){ std::cout << "Cannot create input buffer: " << status << "\n"; return -1; }
        
        auto bufferPartials = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,  nBlocksH*3*256*sizeof(unsigned int), nullptr,  &status);
        if(status != CL_SUCCESS){ std::cout << "Cannot create partial buffer: " << status << "\n"; return -1; }
        
        cl_uchar4 zero = {0, 0, 0, 0};
        status = clEnqueueFillBuffer(queue, bufferPartials, &zero, sizeof(cl_uchar4), 0, nBlocksH*3*256*sizeof(unsigned int), 0, nullptr, nullptr);
        if(status != CL_SUCCESS){ std::cout << "Cannot zero partial buffer: " << status << "\n"; return -1; }

        status = clFinish(queue);
        if(status != CL_SUCCESS){ std::cout << "Cannot finish queue: " << status << "\n"; return -1; }

        auto bufferOutput  = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,  3*256*sizeof(unsigned int), nullptr,  &status);
        if(status != CL_SUCCESS){ std::cout << "Cannot create output buffer: " << status << "\n"; return -1; }

        cl_event evt[2];

        //First kernel of global histograms:
        {
            status = clSetKernelArg(kernel_shared_atomics, 0, sizeof(bufferPartials), &bufferPartials); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 1 arg 0: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_shared_atomics, 1, sizeof(bufferInput),    &bufferInput);    if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 1 arg 1: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_shared_atomics, 2, sizeof(int),            &w);              if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 1 arg 2: " << status << "\n"; return -1; }

            size_t kernel_global_size[2] = {(size_t)block_size, (size_t)nBlocksH*block_size};
            size_t kernel_local_size[2] = {(size_t)block_size, (size_t)block_size};
	        status = clEnqueueNDRangeKernel(queue, kernel_shared_atomics, 2, nullptr, kernel_global_size, kernel_local_size, 0, nullptr, &evt[0]);
            if(status != CL_SUCCESS){ std::cout << "Cannot enqueue kernel 3: " << status << "\n"; return -1; }
        }

        //Second kernel: accumulate partial results:
        {
            status = clSetKernelArg(kernel_accumulate, 0, sizeof(bufferOutput),   &bufferOutput);   if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 2 arg 0: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_accumulate, 1, sizeof(bufferPartials), &bufferPartials); if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 2 arg 1: " << status << "\n"; return -1; }
            status = clSetKernelArg(kernel_accumulate, 2, sizeof(int),            &nBlocksH);       if(status != CL_SUCCESS){ std::cout << "Cannot set kernel 2 arg 2: " << status << "\n"; return -1; }

            size_t kernel_global_size[1] = {(size_t)(3*256)};
	        status = clEnqueueNDRangeKernel(queue, kernel_accumulate, 1, nullptr, kernel_global_size, nullptr, 0, nullptr, &evt[1]);
            if(status != CL_SUCCESS){ std::cout << "Cannot enqueue kernel 4: " << status << "\n"; return -1; }
        }

        std::vector<unsigned int> tmp(3*256);
        
        status = clEnqueueReadBuffer(queue, bufferOutput, CL_TRUE, 0, 3*256*sizeof(unsigned int), tmp.data(), 1, &evt[1], nullptr);
        if(status != CL_SUCCESS){ std::cout << "Cannot read buffer: " << status << "\n"; return -1; }

        cl_ulong t1_0, t1_1;
        status = clGetEventProfilingInfo(evt[0], CL_PROFILING_COMMAND_START, sizeof(t1_0), &t1_0, nullptr);
        status = clGetEventProfilingInfo(evt[1], CL_PROFILING_COMMAND_END,   sizeof(t1_1), &t1_1, nullptr);
        dt2 = (t1_1 - t1_0)*0.001f*0.001f;

        clReleaseEvent(evt[0]);
        clReleaseEvent(evt[1]);

        gpu2.fromLinearMemory(tmp);

        clReleaseMemObject(bufferInput);
        clReleaseMemObject(bufferPartials);
        clReleaseMemObject(bufferOutput);
    }

    clReleaseKernel(kernel_global_atomics);
    clReleaseKernel(kernel_accumulate);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
    
    auto compare = [w, h, &cpu](three_histograms const& histos)
    {
        int mismatches = 0;
        for(int i=0; i<256; ++i)
        {
            if(histos.rh[i] != cpu.rh[i]){ std::cout << "Mismatch: red   at " << i << " : " << histos.rh[i] << " != " << cpu.rh[i] << "\n"; mismatches += 1; }
            if(histos.gh[i] != cpu.gh[i]){ std::cout << "Mismatch: green at " << i << " : " << histos.gh[i] << " != " << cpu.gh[i] << "\n"; mismatches += 1; }
            if(histos.bh[i] != cpu.bh[i]){ std::cout << "Mismatch: blue  at " << i << " : " << histos.bh[i] << " != " << cpu.bh[i] << "\n"; mismatches += 1; }
        }
        return mismatches;
    };

    int mismatches1 = compare(gpu1);
    if     (mismatches1 == 0){ std::cout << "CPU result matches GPU global atomics result.\n"; }
    else if(mismatches1 == 1){ std::cout << "There was 1 mismatch between the CPU and GPU global atomics result.\n"; }
    else                     { std::cout << "There were " << mismatches1 << " mismatches between the CPU and GPU global atomics result.\n"; }

    int mismatches2 = compare(gpu2);
    if     (mismatches2 == 0){ std::cout << "CPU result matches GPU shared atomics result.\n"; }
    else if(mismatches2 == 1){ std::cout << "There was 1 mismatch between the CPU and GPU shared atomics result.\n"; }
    else                     { std::cout << "There were " << mismatches2 << " mismatches between the CPU and GPU shared atomics result.\n"; }

    std::cout << "CPU Computation took:                " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "GPU global atomics computation took: " << dt1  << " ms\n";
    std::cout << "GPU shared atomics computation took: " << dt2 << " ms\n";
    
    auto write_histogram = [](std::string const& filename, three_histograms const& data )
    {
        int w = 800;
        int h = 800;
        std::vector<color> image(w*h);
        color white{255, 255, 255, 255};
        std::fill(image.begin(), image.end(), white);
        auto max_r = *std::max_element(data.rh.begin(), data.rh.end());
        auto max_g = *std::max_element(data.gh.begin(), data.gh.end());
        auto max_b = *std::max_element(data.bh.begin(), data.bh.end());
        auto div = std::max(std::max(max_r, max_g), max_b);

        auto fill_rect = [&](int x0, int y0, int width, int height, color const& c)
        {
            for(int y=y0; y>y0-height; --y)
            {
                for(int x=x0; x<x0+width; ++x)
                {
                    image[y*w+x] = c;
                }
            }
        };

        for(int i=0; i<256; ++i)
        {
            //std::cout << i << "   " << data.rh[i] << " " << data.gh[i] << " " << data.bh[i] << "\n";
            fill_rect(i, 780, 1, data.rh[i]*700/div, color{(unsigned char)i, 0, 0, 255});
            fill_rect(i+256, 780, 1, data.gh[i]*700/div, color{0, (unsigned char)i, 0, 255});
            fill_rect(i+256*2, 780, 1, data.bh[i]*700/div, color{0, 0, (unsigned char)i, 255});
        }
        

        int res = stbi_write_jpg(filename.c_str(), w, h, 4, image.data(), 40);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }else{ std::cout << "Output written to file " << filename << "\n"; }
    };

    write_histogram(output_filename1, cpu);
    write_histogram(output_filename2, gpu1);
    write_histogram(output_filename3, gpu2);

    //free input image
    stbi_image_free(data0);

	return 0;
}
