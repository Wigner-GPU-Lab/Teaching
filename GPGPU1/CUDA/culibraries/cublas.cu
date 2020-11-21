#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <cublas_v2.h>

 int main()
 {
    const int N = 1024;

    std::vector<float> A(N*N);
    std::vector<float> B(N*N);
    std::vector<float> C1(N*N);
    std::vector<float> C2(N*N);

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};

    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(A.begin(), A.end(), gen);
    generate(B.begin(), B.end(), gen);
    std::fill(C1.begin(), C1.end(), 0.0f);
    std::fill(C2.begin(), C2.end(), 0.0f);

    // Initialize cublas:
    cublasHandle_t handle;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    status = cublasCreate(&handle);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot initialize cublas:" << status << "\n"; return -1; }

    cudaError_t cudaStatus = cudaSuccess;

    // Create and set stream, create events:
    cudaStream_t stream;
    cudaStatus = cudaStreamCreate(&stream);
    if( cudaStatus != cudaSuccess ){ std::cout << "Error creating CUDA stream: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    status = cublasSetStream(handle, stream);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot set cublas stream:" << status << "\n"; return -1; }

    cudaEvent_t evt[2];
    for(auto& e : evt)
    {
        auto cudaStatus = cudaEventCreate(&e);
        if(cudaStatus != cudaSuccess){ std::cout << "Error creating event: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    }

    // Allocate device data:
    float* cbA = nullptr;
    float* cbB = nullptr;
    float* cbC = nullptr;
    
    cudaStatus = cudaMalloc((void**)&cbA, N*N*sizeof(float));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for matrix A:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMalloc((void**)&cbB, N*N*sizeof(float));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for matrix B:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMalloc((void**)&cbC, N*N*sizeof(float));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for matrix C:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    // Set matrix data:
    status = cublasSetMatrix(N, N, sizeof(float), A.data(), N, cbA, N);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot upload contents for cublas matrix A:" << status << "\n"; return -1; }

    status = cublasSetMatrix(N, N, sizeof(float), B.data(), N, cbB, N);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot upload contents for cublas matrix B:" << status << "\n"; return -1; }

    cudaStatus = cudaEventRecord(evt[0], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (0): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    // Matrix multiplication funciton: https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    float one = 1.0f;
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &one, cbA, N, cbB, N, &one, cbC, N);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot start cublas matrix multiplication:" << status << "\n"; return -1; }

    cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess ){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaEventRecord(evt[1], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (1): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaDeviceSynchronize();

    float dt = 0.0f; //milliseconds
    cudaStatus = cudaEventElapsedTime(&dt, evt[0], evt[1]);
    if(cudaStatus != cudaSuccess){ std::cout << "Error getting event 0-1 elapsed time: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    std::cout << "Cublas matrix multiplication took " << dt << " ms.\n";

    status = cublasGetMatrix(N, N, sizeof(float), cbC, N, C2.data(), N);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot copy back results for cublas matrix C:" << status << "\n"; return -1; }

    cudaStatus = cudaFree(cbA);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for matrix A:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(cbB);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for matrix B:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(cbC);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for matrix C:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    for(auto& e : evt)
    {
        cudaStatus = cudaEventDestroy(e);
        if(cudaStatus != cudaSuccess){ std::cout << "Error destroying event: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    }

    status = cublasSetStream(handle, 0);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot reset cublas stream:" << status << "\n"; return -1; }

    cudaStatus = cudaStreamDestroy(stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error destroying CUDA stream: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    status = cublasDestroy(handle);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Error shutting down cublas:" << status << "\n"; return -1; }

    return 0;
  }