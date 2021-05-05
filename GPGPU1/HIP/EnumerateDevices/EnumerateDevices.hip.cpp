#include <hip/hip_runtime.h>

#include <iostream> // std::cerr, std::cout
#include <cstdlib>  // std::exit
#include <cstdint>  // int32_t

void checkErr(hipError_t err, const char * msg)
{
    if (err != hipSuccess)
    {
        std::cerr << "ERROR: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        std::exit( err );
    }
}

int main()
{
    int32_t count = -1;
    hipError_t err = hipSuccess;

    err = hipGetDeviceCount(&count);
    checkErr(err, "hipGetDeviceCount");

    std::cout <<
        "Found " <<
        count <<
        " device" <<
        (count > 1 ? "s.\n" : ".\n") <<
        std::endl;

    for(decltype(count) i = 0 ; i < count ; ++i)
    {
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, 0);
        checkErr(err, "hipGetDeviceProperties");

        std::cout << "\t" << prop.name << std::endl;
    }

    return 0;
}
