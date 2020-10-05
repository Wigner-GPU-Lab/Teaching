#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include "cpu_matmul.h"

__global__ void matmul(float* C, float* A, float* B, int N) 
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float sum = 0;
    for(int k=0; k<N; ++k)
    {
        sum += A[y*N+k] * B[k*N+x];
    }
    C[y*N+x] = sum;
}

int main()
{
    const int N = 1024;
    const int block_sz = 16;
    const int n_blocks = N / block_sz;

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
	
	float* pA = nullptr;
    float* pB = nullptr;
    float* pC2 = nullptr;

    cudaEvent_t evt[2];
    for(auto& e : evt){ cudaEventCreate(&e); }

	cudaError_t err = cudaSuccess;
	err = cudaMalloc( (void**)&pA, N*N*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&pB, N*N*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaMalloc( (void**)&pC2, N*N*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( pA, A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice );
    if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaMemcpy( pB, B.data(), N*N*sizeof(float), cudaMemcpyHostToDevice );
	if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    {
        dim3 dimGrid( n_blocks, n_blocks );
        dim3 dimBlock( block_sz, block_sz );
        cudaEventRecord(evt[0]);
        matmul<<<dimGrid, dimBlock>>>(pC2, pA, pB, N);
        err = cudaGetLastError();
	    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
        cudaEventRecord(evt[1]);
    }

	err = cudaMemcpy( C2.data(), pC2, N*N*sizeof(float), cudaMemcpyDeviceToHost );
	if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( pA );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( pB );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( pC2 );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    cudaEventSynchronize(evt[1]);
    float dt = 0.0f;//milliseconds
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    for(auto& e : evt){ cudaEventDestroy(e); }

    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul_naive(C1, A, B, N);
    auto t1 = std::chrono::high_resolution_clock::now();

    const float max_err = 1e-5f;
    auto comparator = [max_err](float l, float r){ return std::abs(l-r) < max_err; };
    
    for(int i=0; i<N*N; ++i)
	{
        if( !comparator(C1[i], C2[i]) )
        {
            std::cout << "[" << i << "] : " << C1[i] << "   " << C2[i] << " absolute error: " << std::abs(C1[i]-C2[i]) << "\n";
        }
    }

    if(std::equal(C1.begin(), C1.end(), C2.begin(), comparator))
	{
		std::cout << "Success\n";
	}
	else{ std::cout << "Mismatch between CPU and GPU results.\n"; }
    
    std::cout << "CPU Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "GPU Computation took: " << dt << " ms.\n";
	return 0;
}
