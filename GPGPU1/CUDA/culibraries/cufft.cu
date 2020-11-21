#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <cufft.h>

 int main()
 {
    const int N = 1024;

    std::vector<float>        A(N);
    std::vector<cufftComplex> C(N);

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};

    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(A.begin(), A.end(), gen);
    std::fill(C.begin(), C.end(), cufftComplex{0.0f, 0.0f});

    cudaError_t cudaStatus = cudaSuccess;

    // Create and set stream, create events:
    cudaStream_t stream;
    cudaStatus = cudaStreamCreate(&stream);
    if( cudaStatus != cudaSuccess ){ std::cout << "Error creating CUDA stream: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaEvent_t evt[3];
    for(auto& e : evt)
    {
        auto cudaStatus = cudaEventCreate(&e);
        if(cudaStatus != cudaSuccess){ std::cout << "Error creating event: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    }

    cufftResult result = CUFFT_SUCCESS;

    // Create FFT Plans:
    cufftHandle   plan_fwd, plan_bwd;
    result = cufftPlan1d(&plan_fwd, N, CUFFT_R2C, 1);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot create fwd cufft plan: " << result << "\n"; return -1; }

    result = cufftSetStream(plan_fwd, stream);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot set stream for fwd cufft: " << result << "\n"; return -1; }

    result = cufftPlan1d(&plan_bwd, N/2+1, CUFFT_C2C, 1);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot create bwd cufft plan: " << result << "\n"; return -1; }

    result = cufftSetStream(plan_bwd, stream);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot set stream for bwd cufft: " << result << "\n"; return -1; }
    

    // Allocate device data:
    float*        bA = nullptr;
    cufftComplex* bB = nullptr;
    cufftComplex* bC = nullptr;
    
    cudaStatus = cudaMalloc((void**)&bA, N*sizeof(float));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for float array A:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMalloc((void**)&bB, sizeof(cufftComplex)*(N/2+1));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for complex array B:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMalloc((void**)&bC, sizeof(cufftComplex)*(N/2+1));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for complex array C:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMemcpy(bA, A.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot copy data to device memory:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaEventRecord(evt[0], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (0): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    result = cufftExecR2C(plan_fwd, (cufftReal*)bA, bB);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot execute R2C fft: " << result << "\n"; return -1; }

    cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess ){ std::cout << "CUDA error in kernel call R2C: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaEventRecord(evt[1], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (1): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    result = cufftExecC2C(plan_bwd, bB, bC, CUFFT_INVERSE);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot execute inverse C2C fft: " << result << "\n"; return -1; }

    cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess ){ std::cout << "CUDA error in kernel call R2C: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaEventRecord(evt[2], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (2): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaDeviceSynchronize();

    float dt_fwd = 0.0f, dt_bwd; //milliseconds
    cudaStatus = cudaEventElapsedTime(&dt_fwd, evt[0], evt[1]);
    if(cudaStatus != cudaSuccess){ std::cout << "Error getting event 0-1 elapsed time: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    cudaStatus = cudaEventElapsedTime(&dt_bwd, evt[1], evt[2]);
    if(cudaStatus != cudaSuccess){ std::cout << "Error getting event 1-2 elapsed time: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    std::cout << "Forward FFT took: " << dt_fwd << " ms.\n";
    std::cout << "Backward FFT took: " << dt_bwd << " ms.\n";

    // Copy device memory to host
    cudaStatus = cudaMemcpy(C.data(), bC, sizeof(cufftComplex)*(N/2+1), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess){ std::cout << "Error copying data back: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(bA);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for array A: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(bB);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for array B: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(bC);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for array C: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    for(auto& e : evt)
    {
        cudaStatus = cudaEventDestroy(e);
        if(cudaStatus != cudaSuccess){ std::cout << "Error destroying event: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    }

    result = cufftSetStream(plan_fwd, 0);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot reset cufft stream: " << result << "\n"; return -1; }
    result = cufftSetStream(plan_bwd, 0);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot reset cufft stream: " << result << "\n"; return -1; }

    cudaStatus = cudaStreamDestroy(stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error destroying CUDA stream: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    result = cufftDestroy(plan_fwd);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot destroy cufft plan: " << result << "\n"; return -1; }

    result = cufftDestroy(plan_bwd);
    if(result != CUFFT_SUCCESS){ std::cout << "Cannot destroy cufft plan: " << result << "\n"; return -1; }

    for(int i=0; i<(int)C.size(); ++i)
    {
        std::cout << i << "   " << A[i] << "   " << C[i].x << "   " << C[i].y << "\n";
    }

    return 0;
  }