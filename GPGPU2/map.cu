#include <vector>
#include <iostream>

template<typename F, typename T>
__global__ void map(F f, T* src, T* dst)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    dst[id] = f(src[id]); 
}
 
int main( )
{
    using T = double;
    using P = T*;

    // Size of vectors
    int n = 16;

    // Size, in bytes, of each vector
    size_t sz = n*sizeof(T);
 
    // Host vectors
    std::vector<T> h_src(n);
    std::vector<T> h_dst(n);
   
    // Device vectors
    P d_src, d_dst;

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_src, sz);
    cudaMalloc(&d_dst, sz);
 
    // Initialize vectors on host
    for(int i = 0; i < n; i++ )
    {
        h_src[i] = i*2.0 + 1.0;
        h_dst[i] = 0;
        
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_src, h_src.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy( d_dst, h_dst.data(), sz, cudaMemcpyHostToDevice);
 
    // Number of threads in each thread block
    int blockSize = 4;
 
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    auto sq = [] __device__ (auto const& x){ return x*x; };
    
    map<<<gridSize, blockSize>>>(sq, d_src, d_dst);
 
    // Copy array back to host
    cudaMemcpy( h_dst.data(), d_dst, sz, cudaMemcpyDeviceToHost );
 
    for(int i=0; i<n; i++)
    {
        std::cout << "result[" << i << "] = " << h_dst[i] << "\n";
    }

    // Release device memory
    cudaFree(d_src);
    cudaFree(d_dst);
 
    return 0;
}
