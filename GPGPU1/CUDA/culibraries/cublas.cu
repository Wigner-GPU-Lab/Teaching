#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <cublas_v2.h>
#include "cpu_matmul.h"

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
    // All matrixes need to be transposed, including the result C, but since C^T = (AB)^T = (B^T A^T), we dont need to transpose the input matrices, just change their order:
    float one = 1.0f;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, cbB, N, cbA, N, &one, cbC, N);
    if(status != CUBLAS_STATUS_SUCCESS){ std::cout << "Cannot start cublas matrix multiplication:" << status << "\n"; return -1; }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess ){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaEventRecord(evt[1], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (1): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaDeviceSynchronize();

    float dt = 0.0f; //milliseconds
    cudaStatus = cudaEventElapsedTime(&dt, evt[0], evt[1]);
    if(cudaStatus != cudaSuccess){ std::cout << "Error getting event 0-1 elapsed time: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    
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

    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_matmul_improved(C1, A, B, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    const float max_err = 1e-5f;
    auto comparator = [max_err](float l, float r){ return std::abs(l-r) < max_err; };
    
    for(int i=0; i<N*N; ++i)
	{
        if( !comparator(C1[i], C2[i]) )
        {
            std::cout << "C1 vs C2 [" << i << "] : " << C1[i] << "   " << C2[i] << " absolute error: " << std::abs(C1[i]-C2[i]) << "\n";
        }
    }

    if( std::equal(C1.begin(), C1.end(), C2.begin(), comparator) )
	{
		std::cout << "GPU improved matches CPU naive.\n";
	}
	else
	{
		std::cout << "Mismatch in CPU and GPU results.\n";
	}
    
    std::cout << "CPU improved Computation took:     " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0f << " ms\n";
    std::cout << "Cublas matrix multiplication took: " << dt << " ms.\n";

    return 0;
}