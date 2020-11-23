#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

static const double pi = 3.1415926535897932384623;

static const unsigned int blockSize = 512;

__global__ void initialize(unsigned long long* seeds,
                           unsigned long long* subseqs,
                           unsigned long long* offsets,
                           curandState_t*      states)
{
    int idx1 = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = 2 * idx1;
    
    curand_init(seeds[idx2 + 0], subseqs[idx2 + 0], offsets[idx2 + 0], &states[idx2 + 0]);
    curand_init(seeds[idx2 + 1], subseqs[idx2 + 1], offsets[idx2 + 1], &states[idx2 + 1]);
}

__device__ void warpReduce(volatile double* tmp, int tid)
{
    tmp[tid] += tmp[tid + 32];
    tmp[tid] += tmp[tid + 16];
    tmp[tid] += tmp[tid + 8];
    tmp[tid] += tmp[tid + 4];
    tmp[tid] += tmp[tid + 2];
    tmp[tid] += tmp[tid + 1];
}

__global__ void reduce(double* dst, curandState_t* states, unsigned long long n, unsigned long long N) 
{
    extern __shared__ double tmp[];

    unsigned int tid = threadIdx.x;
    int idx1 = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = 2 * idx1;

    curandState stateX = states[idx2+0];
    curandState stateY = states[idx2+1];
    double acc = 0.0;
    int i = 0;
    while(i < n)
    {
        double part = 0.0;
        double x = curand_uniform_double(&stateX) * 2.0 - 1.0;
        double y = curand_uniform_double(&stateY) * 2.0 - 1.0;
        if(x*x+y*y <= 1.0)
        {
            part = 1.0;
        }
        acc += part;
        i += 1;
    }
    tmp[tid] = acc / N;
    __syncthreads();
    states[idx2+0] = stateX;
    states[idx2+1] = stateY;
    
    // do reduction in shared mem, no loop, it was unrolled
    if(tid < 256){ tmp[tid] += tmp[tid + 256]; } __syncthreads();
    if(tid < 128){ tmp[tid] += tmp[tid + 128]; } __syncthreads();
    if(tid <  64){ tmp[tid] += tmp[tid +  64]; } __syncthreads();
    if(tid <  32){ warpReduce(tmp, tid); }
    
    // write result for this block to global mem
    if(tid == 0){ dst[blockIdx.x] = tmp[0]; }
}

 int main()
 {
    const int nthreads = 4096;
    const unsigned long long N = (unsigned long long)1 << (unsigned long long)36;
    std::cout << "Number of Monte-Carlo samples: " << N << " = 2^" << std::log2(N) << "\n";

    std::vector<unsigned long long> vSeeds  (nthreads*2);
    std::vector<unsigned long long> vSubseqs(nthreads*2);
    std::vector<unsigned long long> vOffsets(nthreads*2);

    std::vector<double> vPartialResults(nthreads / blockSize);

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_int_distribution<unsigned long long> dist{0};//from 0 to max representable

    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(vSeeds.begin(),   vSeeds.end(),   gen);
    generate(vSubseqs.begin(), vSubseqs.end(), gen);
    generate(vOffsets.begin(), vOffsets.end(), gen);
    std::fill(vPartialResults.begin(), vPartialResults.end(), 0.0);

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

    // Allocate device data:
    curandState_t*      bStates  = nullptr;
    unsigned long long* bSeeds   = nullptr;
    unsigned long long* bSubseqs = nullptr;
    unsigned long long* bOffsets = nullptr;
    double*             bPartialResults = nullptr;
    
    cudaStatus = cudaMalloc((void**)&bStates, nthreads*2*sizeof(curandState_t));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for states:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMalloc((void**)&bSeeds, vSeeds.size()*sizeof(unsigned long long));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for seeds:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMalloc((void**)&bSubseqs, vSubseqs.size()*sizeof(unsigned long long));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for subseqs:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
  
    cudaStatus = cudaMalloc((void**)&bOffsets, vOffsets.size()*sizeof(unsigned long long));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for offsets:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
  
    cudaStatus = cudaMalloc((void**)&bPartialResults, vPartialResults.size()*sizeof(double));
    if(cudaStatus != cudaSuccess){ std::cout << "Cannot allocate device memory for partial results:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    // Copy data to device:
    cudaStatus = cudaMemcpy(bSeeds, vSeeds.data(), vSeeds.size()*sizeof(unsigned long long), cudaMemcpyHostToDevice );
    if(cudaStatus != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    
    cudaStatus = cudaMemcpy(bSubseqs, vSubseqs.data(), vSubseqs.size()*sizeof(unsigned long long), cudaMemcpyHostToDevice );
    if(cudaStatus != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    
    cudaStatus = cudaMemcpy(bOffsets, vOffsets.data(), vOffsets.size()*sizeof(unsigned long long), cudaMemcpyHostToDevice );
    if(cudaStatus != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    
    // Initialize random number generators:
    cudaStatus = cudaEventRecord(evt[0], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (0): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    {
        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);
        initialize<<<dimGrid, dimBlock, 0, stream>>>(bSeeds, bSubseqs, bOffsets, bStates);
        cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess){ std::cout << "CUDA error in kernel call 1: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    }
    
    cudaStatus = cudaEventRecord(evt[1], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (1): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    {
        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);
        size_t shared_mem_size = blockSize * sizeof(double);
        reduce<<<dimGrid, dimBlock, shared_mem_size, stream>>>(bPartialResults, bStates, N/nthreads, N);
        cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess){ std::cout << "CUDA error in kernel call 2: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    }

    cudaStatus = cudaEventRecord(evt[2], stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error recording event (2): " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaMemcpy(vPartialResults.data(), bPartialResults, vPartialResults.size()*sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    
    float result = std::accumulate(vPartialResults.begin(), vPartialResults.end(), 0.0)*4.0f;
    std::cout.precision(16);
    std::cout << "Result:    " << result << "\n";
    std::cout << "Reference: " << pi     << "\n";
    std::cout.precision(6);

    cudaDeviceSynchronize();

    float dt1 = 0.0f, dt2 = 0.0f; //milliseconds
    cudaStatus = cudaEventElapsedTime(&dt1, evt[0], evt[1]);
    if(cudaStatus != cudaSuccess){ std::cout << "Error getting event 0-1 elapsed time: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaEventElapsedTime(&dt2, evt[1], evt[2]);
    if(cudaStatus != cudaSuccess){ std::cout << "Error getting event 1-2 elapsed time: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    
    std::cout << "Initialization took: " << dt1 << " milliseconds.\n";
    std::cout << "Computation took:    " << dt2 << " milliseconds.\n";
    
    cudaStatus = cudaFree(bStates);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for states:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(bSeeds);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for seeds:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(bSubseqs);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for subseqs:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(bOffsets);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for offsets:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    cudaStatus = cudaFree(bPartialResults);
    if(cudaStatus != cudaSuccess){ std::cout << "Error freeing device memory for partial results:" << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    for(auto& e : evt)
    {
        cudaStatus = cudaEventDestroy(e);
        if(cudaStatus != cudaSuccess){ std::cout << "Error destroying event: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }
    }

    cudaStatus = cudaStreamDestroy(stream);
    if(cudaStatus != cudaSuccess){ std::cout << "Error destroying CUDA stream: " << cudaGetErrorString(cudaStatus) << "\n"; return -1; }

    return 0;
  }