#include <OpenCL-C-API-minimal.hpp>


int main()
{
    cl_platform_id platform = NULL;
    auto status = clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device = NULL;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    auto context = clCreateContext(cps, 1, &device, 0, 0, &status);

    auto queue = clCreateCommandQueueWithProperties(context, device, nullptr, &status);

    std::ifstream file(kernel_location);
    std::string source(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    size_t      sourceSize = source.size();
    const char* sourcePtr = source.c_str();
    auto program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &status);

    status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (status != CL_SUCCESS)
    {
        size_t len = 0;
        status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::unique_ptr<char[]> log = std::make_unique<char[]>(len);
        status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.get(), nullptr);
        printf("%s\n", log.get());
        system("PAUSE");
    }

    auto kernel = clCreateKernel(program, "squarer", &status);

    std::array<float, 8> A{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    std::array<float, 8> B{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    auto buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(float), A.data(), &status);
    auto buffer_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(float), B.data(), &status);

    status = clSetKernelArg(kernel, 0, sizeof(buffer_in), &buffer_in);
    status = clSetKernelArg(kernel, 1, sizeof(buffer_out), &buffer_out);

    size_t thread_count = A.size();
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &thread_count, nullptr, 0, nullptr, nullptr);
    status = clEnqueueReadBuffer(queue, buffer_out, false, 0, B.size() * sizeof(float), B.data(), 0, nullptr, nullptr);
    status = clFinish(queue);

    std::for_each(B.begin(), B.end(), [](float x) { std::cout << x << "\n"; });

    return 0;
}
