#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

__device__ void warpReduce(volatile float* tmp, int tid)
{
    tmp[tid] += tmp[tid + 32];
    tmp[tid] += tmp[tid + 16];
    tmp[tid] += tmp[tid + 8];
    tmp[tid] += tmp[tid + 4];
    tmp[tid] += tmp[tid + 2];
    tmp[tid] += tmp[tid + 1];
}

__global__ void reduce(float* dst, float* src, int n) 
{
    extern __shared__ float tmp[];
    static const unsigned int blockSize = 512;

    // each thread loads two elements from global, adds them
    // and stores the result to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    tmp[tid] = 0;
    while(i < n){ tmp[tid] += src[i] + src[i+blockSize]; i += gridSize; }
    __syncthreads();
    
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
    //const int divisor1 = reduce_version <= 3 ? 1 : 2;// = 1, but = 2 for kernels 4, and above
    //const int divisor2 = reduce_version <= 5 ? divisor1 : 1;// = divisor1, but = 1 for kernel6 and 7
    
    const size_t extra = 2;
    const size_t sz = 512*512*extra;
    const size_t block_sz = 512;
    const int    n_blocks = sz / extra / block_sz;

	std::vector<float> A(sz);
	std::vector<float> B(n_blocks);

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};

    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(A.begin(), A.end(), gen);
	
	float* src = nullptr;
    float* dst = nullptr;
    float* res = nullptr;
    float  gpu_sum = 0.0f;

    cudaEvent_t evt[4];
    for(auto& e : evt){ cudaEventCreate(&e); }

	cudaError_t err = cudaSuccess;
	err = cudaMalloc( (void**)&src, sz*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&dst, n_blocks*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaMalloc( (void**)&res, 1*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( src, A.data(), sz*sizeof(float), cudaMemcpyHostToDevice );
	if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    {
        dim3 dimGrid( n_blocks/2, 1 );//<<<<<<<<<<divide by 2!
        dim3 dimBlock( block_sz, 1 );
        size_t shared_mem_size = block_sz*sizeof(float);
        cudaEventRecord(evt[0]);
        reduce<<<dimGrid, dimBlock, shared_mem_size>>>(dst, src, dimGrid.x*dimBlock.x*2*extra);//<<<<<mul by 2
        err = cudaGetLastError();
	    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call 1: " << cudaGetErrorString(err) << "\n"; return -1; }
        cudaEventRecord(evt[1]);
    }
	
    {
        dim3 dimGrid( 1, 1 );
        dim3 dimBlock( block_sz, 1 );//<<<<<<<div by 1!
        size_t shared_mem_size = block_sz*sizeof(float);
        cudaEventRecord(evt[2]);
        reduce<<<dimGrid, dimBlock, shared_mem_size>>>(res, dst, dimGrid.x*dimBlock.x);
        err = cudaGetLastError();
	    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call 2: " << cudaGetErrorString(err) << "\n"; return -1; }
        cudaEventRecord(evt[3]);
    }

	err = cudaMemcpy( &gpu_sum, res, sizeof(float), cudaMemcpyDeviceToHost );
	if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( src );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( dst );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( res );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    cudaEventSynchronize(evt[3]);
    float dt1 = 0.0f, dt2 = 0.0f;//milliseconds
    cudaEventElapsedTime(&dt1, evt[0], evt[1]);
    cudaEventElapsedTime(&dt2, evt[2], evt[3]);

    for(auto& e : evt){ cudaEventDestroy(e); }

    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_sum = std::accumulate(A.begin(), A.end(), 0.0f);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout.precision(10);
    std::cout << "cpu_sum = " << cpu_sum << "\n";
    std::cout << "gpu_sum = " << gpu_sum << "\n";

    float rel_err = std::abs((cpu_sum - gpu_sum) / cpu_sum);
    std::cout << "Relative error is: " << rel_err << "\n";
	if( rel_err < 2e-4 )
	{
		std::cout << "Success.\n";
	}
	else
	{
		std::cout << "Mismatch in CPU and GPU result.\n";
    }
    
    std::cout << "CPU Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "GPU Computation took: " << dt1 << " + " << dt2 << " = " << dt1+dt2 << " ms.\n";
	return 0;
}
