#include <vector>
#include <iostream>

template<typename F, typename T>
__global__ void zip(F f, T* src1, T* src2, T* dst)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    dst[id] = f(src1[id], src2[id]); 
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
    std::vector<T> h_src1(n);
    std::vector<T> h_src2(n);
    std::vector<T> h_dst(n);
   
    // Device vectors
    P d_src1, d_src2, d_dst;

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_src1, sz);
    cudaMalloc(&d_src2, sz);
    cudaMalloc(&d_dst, sz);
 
    // Initialize vectors on host
    for(int i = 0; i < n; i++ )
    {
        h_src1[i] = i*1.0;
        h_src2[i] = (i+1)*1.0;
        h_dst[i] = 0;
        
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_src1, h_src1.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy( d_src2, h_src2.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy( d_dst, h_dst.data(), sz, cudaMemcpyHostToDevice);
 
    // Number of threads in each thread block
    int blockSize = 4;
 
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    auto sqsum = [] __device__ (auto const& x, auto const& y){ return x*x + y*y; };
    
    zip<<<gridSize, blockSize>>>(sqsum, d_src1, d_src2, d_dst);
 
    // Copy array back to host
    cudaMemcpy( h_dst.data(), d_dst, sz, cudaMemcpyDeviceToHost );
 
    for(int i=0; i<n; i++)
    {
        std::cout << "result[" << i << "] = " << h_dst[i] << "\n";
    }

    // Release device memory
    cudaFree(d_src1);
    cudaFree(d_src2);
    cudaFree(d_dst);
 
    return 0;
}
