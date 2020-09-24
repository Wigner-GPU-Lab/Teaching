#include <vector>
#include <algorithm>
#include <iostream>

int main()
{
	int dev_count = 0;

    cudaError_t err = cudaSuccess;
    err = cudaGetDeviceCount(&dev_count);
    if( err != cudaSuccess){ std::cout << "Error getting device count: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    std::cout << "There are " << dev_count << " device(s)\n";
    std::cout << "--------------------------------------------------------------------------\n";
    
    for(int d = 0; d < dev_count; ++d)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, d);
        if( err != cudaSuccess){ std::cout << "Error getting device properties for device " << d << ": " << cudaGetErrorString(err) << "\n"; continue; }
        
        std::cout << "Device number:               " << d         << "\n";
        std::cout << "Device name:                 " << prop.name << "\n";
        std::cout << "Device global memory:        " << prop.totalGlobalMem/1024/1024 << "MiB\n";
        std::cout << "Device warp size:            " << prop.warpSize << "\n";
        std::cout << "Device max grid size:        " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "\n";
        
        auto attribute = [d](auto attr)
        {
            int value = 0;
            auto err = cudaDeviceGetAttribute(&value, attr, d);
            if( err != cudaSuccess){ std::cout << "Error getting device attribute '" << attr << "': " << cudaGetErrorString(err) << "\n"; return -1; };
            return value;
        };

        std::cout << "Max threads per block:       " << attribute(cudaDevAttrMaxThreadsPerBlock) << "\n";
        std::cout << "Max shared memory per block: " << attribute(cudaDevAttrMaxSharedMemoryPerBlock) << " bytes\n";
        std::cout << "--------------------------------------------------------------------------\n";
    }

	return 0;
}