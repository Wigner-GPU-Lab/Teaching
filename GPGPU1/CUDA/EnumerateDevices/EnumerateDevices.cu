#include <cuda.h>

#include <stdio.h>  // printf

void checkErr(cudaError_t err, const char * name)
{
    if (err != cudaSuccess)
    {
        printf("ERROR: %s (%i)\n", name, err);
        exit( err );
    }
}

int main()
{
    cudaError_t err = cudaSuccess;
    int numDevices = 0;

    err = cudaGetDeviceCount(&numDevices);
    checkErr(err, "cudaGetDeviceCount()");

    if (numDevices == 0)
    {
        printf("No CUDA devices detected.\n");
        exit( -1 );
    }
    printf("Found %u device(s)\n", numDevices);
    fflush(NULL);

    for (int i = 0; i < numDevices; ++i)
    {
        cudaDeviceProp deviceProps;
        err = cudaGetDeviceProperties(&deviceProps, i);
        checkErr(err, "cudaGetDeviceProperties()");

        printf("\t%s\n", deviceProps.name);
    }

    return 0;
}
