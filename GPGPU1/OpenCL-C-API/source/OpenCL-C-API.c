#include <OpenCL-C-API.h>

// OpenCL error handling function
void checkErr( cl_int err, const char * name )
{
    if ( err != CL_SUCCESS )
    {
        printf_s("ERROR: %s (%i)\n", name, err);

        exit( err );
    }
}

// Probe for available OpenCL platforms
void probe_platforms()
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );
    checkErr( CL_err, "clGetPlatformIDs(numPlatforms)" );

    if ( numPlatforms > 0 )
    {
        cl_platform_id* platforms = (cl_platform_id*)malloc( numPlatforms * sizeof( cl_platform_id ) );
        CL_err = clGetPlatformIDs( numPlatforms, platforms, NULL );
        checkErr( CL_err, "clGetPlatformIDs(platforms)" );

        printf_s( "%u platform(s) found:\n", numPlatforms );

        for ( cl_uint i = 0; i < numPlatforms; ++i )
        {
            char pbuf[100];
            CL_err = clGetPlatformInfo( platforms[i], CL_PLATFORM_VENDOR, sizeof( pbuf ), pbuf, NULL );
            checkErr( CL_err, "clGetPlatformInfo(CL_PLATFORM_VENDOR)" );

            printf_s( "\t%s\n", pbuf );
        }

        free( platforms );
    }
    else
    {
        printf_s( "No OpenCL platform detected.\n" );

        exit( -1 );
    }
}

// Select platform with most DP capable devices
cl_platform_id select_platform()
{
    cl_int CL_err = CL_SUCCESS;
    cl_platform_id result = NULL;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );
    checkErr( CL_err, "clGetPlatformIDs(numPlatforms)" );

    cl_platform_id* platforms = (cl_platform_id*)malloc( numPlatforms * sizeof( cl_platform_id ) );
    CL_err = clGetPlatformIDs( numPlatforms, platforms, NULL );
    checkErr( CL_err, "clGetPlatformIDs(platforms)" );

    cl_uint max_count = 0;

    for ( cl_uint i = 0; i < numPlatforms; ++i )
    {
        cl_uint count = count_dp_capable_devices( platforms[i] );

        if ( count > max_count )
        {
            max_count = count;
            result = platforms[i];
        }
    }

    if ( result == NULL )
    {
        printf_s( "No double precision capable HW detected.\n" );

        exit( -1 );
    }

    free( platforms );

    return result;
}

// Test device for DP capability
cl_bool is_device_DP_capable( cl_device_id device )
{
    cl_int CL_err = CL_SUCCESS;
    char pbuf[1024];

    CL_err = clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, sizeof( pbuf ), pbuf, NULL );
    checkErr( CL_err, "clGetDeviceInfo(CL_DEVICE_EXTENSIONS)" );

    return strstr( pbuf, "cl_khr_fp64" ) || strstr( pbuf, "cl_amd_fp64" );
}

// Count devices with DP capability
unsigned int count_dp_capable_devices( cl_platform_id platform )
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint result = 0;
    cl_device_id* devices = NULL;
    cl_uint numDevices = 0;

    CL_err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 0, 0, &numDevices );
    checkErr( CL_err, "clGetDeviceIDs(numDevices)" );
    devices = (cl_device_id*)malloc( numDevices * sizeof( cl_device_id ) );
    CL_err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, numDevices, devices, 0 );
    checkErr( CL_err, "clGetDeviceIDs(devices)" );

    for ( cl_uint i = 0; i < numDevices; ++i )
        if ( is_device_DP_capable( devices[i] ) )
            ++result;

    for (cl_uint i = 0; i < numDevices; ++i) clReleaseDevice(devices[i]);
    free( devices );

    return result;
}

// Select devices with DP capability
cl_device_id* select_devices( cl_platform_id platform )
{
    cl_int CL_err = CL_SUCCESS;
    cl_device_id* result = NULL;
    cl_device_id* devices = NULL;
    cl_uint numDevices = 0;
    
    CL_err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 0, 0, &numDevices );
    checkErr( CL_err, "clGetDeviceIDs(numDevices)" );
    devices = (cl_device_id*)malloc( numDevices * sizeof( cl_device_id ) );
    CL_err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, numDevices, devices, 0 );
    checkErr( CL_err, "clGetDeviceIDs(devices)" );

    cl_uint numDevicesDP = count_dp_capable_devices( platform );
    result = (cl_device_id*)malloc(numDevicesDP * sizeof( cl_device_id ) );

    cl_uint counter = 0;

    for (cl_uint i = 0; i < numDevices; ++i)
        if (is_device_DP_capable(devices[i]))
            result[counter++] = devices[i];
        //else
        //    clReleaseDevice(devices[i]);

    free( devices );

    return result;
}

// Create standard context
cl_context create_standard_context( cl_device_id* devices )
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint count = 0;
    cl_context result = NULL;
    cl_platform_id platform = NULL;

    CL_err = clGetDeviceInfo( devices[0], CL_DEVICE_PLATFORM, sizeof( cl_platform_id ), &platform, NULL );
    checkErr( CL_err, "clGetDeviceInfo(CL_DEVICE_PLATFORM)" );

    count = count_dp_capable_devices( platform );

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

    result = clCreateContext( cps, count, devices, NULL, NULL, &CL_err );
    checkErr( CL_err, "clCreateContext()" );

    return result;
}

// Load kernel file
char* load_program_file( const char* filename )
{
    long int size = 0;
    size_t res = 0;
    char* src = NULL;
    FILE* file = NULL;
    errno_t err = 0;

    err = fopen_s( &file, filename, "rb");

    if ( !file )
    {
        printf_s( "Failed to open file %s\n", filename );

        exit( EXIT_FAILURE );
    }

    if ( fseek( file, 0, SEEK_END ) )
    {
        fclose( file );
        return NULL;
    }

    size = ftell( file );
    if ( size == 0 )
    {
        fclose( file );
        return NULL;
    }

    rewind( file );

    src = (char *)calloc( size + 1, sizeof( char ) );
    if ( !src )
    {
        src = NULL;
        fclose( file );
        return src;
    }

    res = fread( src, 1u, sizeof( char ) * size, file);
    if ( res != sizeof( char ) * size )
    {
        fclose( file );
        free( src );

        return src;
    }

    src[size] = '\0'; /* NULL terminated */
    fclose( file );

    return src;
}

// Build program file
cl_program build_program_source( cl_context context, const char* source )
{
    cl_int CL_err = CL_SUCCESS;
    cl_program result = NULL;

    cl_uint numDevices = 0;
    cl_device_id* devices = NULL;

    const size_t length = strnlen_s(source, UINT_MAX);

    result = clCreateProgramWithSource( context, 1, &source, &length, &CL_err );
    checkErr( CL_err, "clCreateProgramWithSource()" );

    CL_err = clGetContextInfo( context, CL_CONTEXT_NUM_DEVICES, sizeof( cl_uint ), &numDevices, NULL );
    checkErr( CL_err, "clGetContextInfo(CL_CONTEXT_NUM_DEVICES)" );
    devices = (cl_device_id*)malloc( numDevices * sizeof( cl_device_id ) );
    CL_err = clGetContextInfo( context, CL_CONTEXT_DEVICES, numDevices * sizeof( cl_device_id ), devices, NULL );
    checkErr( CL_err, "clGetContextInfo(CL_CONTEXT_DEVICES)" );

    // Warnings will be treated like errors, this is useful for debug
    char build_params[] = { "-Werror" };
    CL_err = clBuildProgram( result, numDevices, devices, build_params, NULL, NULL );

    if ( CL_err != CL_SUCCESS )
    {
        size_t len = 0;
        char *buffer;

        CL_err = clGetProgramBuildInfo( result, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len );
        checkErr( CL_err, "clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG)" );

        buffer = calloc( len, sizeof( char ) );

        clGetProgramBuildInfo( result, devices[0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL );

        fprintf( stderr, "%s\n", buffer );

        free( buffer );

        exit( CL_err );
    }

    return result;
}

// Obtain kernel from program
cl_kernel obtain_kernel( cl_program program, const char* name )
{
    cl_int CL_err = CL_SUCCESS;
    cl_kernel result = NULL;

    result = clCreateKernel( program, name, &CL_err );
    checkErr( CL_err, "clCreateKernel()" );

    return result;
}

int main()
{
    probe_platforms();

    cl_platform_id platform = select_platform();

    cl_device_id* devices = select_devices( platform );

    cl_context context = create_standard_context( devices );

    char* src = load_program_file( kernel_location );

    cl_program program = build_program_source( context, src );

    cl_kernel kernel = obtain_kernel( program, "vecAdd" );

    cl_int CL_err = CL_SUCCESS;
    const size_t chainlength = 1048576;

    cl_double a = 2.0;
    cl_double* x = (cl_double*)malloc( chainlength * sizeof( cl_double ) );
    cl_double* y = (cl_double*)malloc( chainlength * sizeof( cl_double ) );

    for (size_t i = 0; i < chainlength; ++i) x[i] = y[i] = 1;
    
    cl_mem buf_x = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, chainlength * sizeof( cl_double ), x, &CL_err );
    checkErr( CL_err, "clCreateBuffer(buf_x)" );
    cl_mem buf_y = clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, chainlength * sizeof( cl_double ), y, &CL_err );
    checkErr(CL_err, "clCreateBuffer(buf_y)");

    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;

    cl_command_queue queue = clCreateCommandQueue( context, devices[0], props, &CL_err );
    checkErr( CL_err, "clCreateCommandQueue()" );

    CL_err = clSetKernelArg( kernel, 0, sizeof( cl_double ), &a );
    checkErr( CL_err, "clSetKernelArg(a)" );
    CL_err = clSetKernelArg( kernel, 1, sizeof( cl_mem ), &buf_x );
    checkErr( CL_err, "clSetKernelArg(buf_x)" );
    CL_err = clSetKernelArg( kernel, 2, sizeof( cl_mem ), &buf_y );
    checkErr( CL_err, "clSetKernelArg(buf_y)" );

    cl_event kernel_event;

    CL_err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, &chainlength, NULL, 0, NULL, &kernel_event );
    checkErr( CL_err, "clEnqueueNDRangeKernel(kernel)" );

    clWaitForEvents( 1, &kernel_event );

    cl_ulong exec_start, exec_end;

    CL_err = clGetEventProfilingInfo( kernel_event, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &exec_start, NULL );
    checkErr( CL_err, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_START)" );
    CL_err = clGetEventProfilingInfo( kernel_event, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &exec_end, NULL );
    checkErr( CL_err, "clGetEventProfilingInfo(CL_PROFILING_COMMAND_END)" );

    printf_s( "Kernel execution took: %llu nanoseconds\n", exec_end - exec_start );

    CL_err = clEnqueueReadBuffer( queue, buf_y, CL_TRUE, 0, chainlength * sizeof( cl_double ), y, 0, NULL, NULL );
    checkErr( CL_err, "clEnqueueReadBuffer(buf_y)" );

    // Validate results
    int fail = 0;
    for (size_t i = 0; i < chainlength; ++i) fail |= (fabs(y[i] - 3.0) > 1e-10);

    if (fail)
    {
        printf_s("Validation failed\n");
        exit(EXIT_FAILURE);
    }

    // Release OpenCL resources
    cl_uint count = 0;
    CL_err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &count, NULL);

    for (cl_uint i = 0; i < count; ++i)
        clReleaseDevice(devices[i]);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(buf_x);
    clReleaseMemObject(buf_y);
    clReleaseCommandQueue(queue);
    clReleaseEvent(kernel_event);

    // Free host-side memory
    free(x);
    free(y);

    return EXIT_SUCCESS;
}