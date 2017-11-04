#include <vector>
#include <iostream>

template<typename F, typename T>
__global__ void reduce(F f, T* src, T* dst)
{
    extern __shared__ T tmp[];

    // each thread loads two elements from global and reduces them to shared memory
    auto l = threadIdx.x;
    auto i = blockIdx.x * (2*blockDim.x) + threadIdx.x;

    tmp[l] = f( src[i], src[i+blockDim.x] );
    
    __syncthreads();

    // do reduction in shared mem
    for(auto s=blockDim.x/2; s > 0; s >>= 1)
    {
        if(l < s)
        {
            tmp[l] = f(tmp[l], tmp[l + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(l == 0){ dst[blockIdx.x] = tmp[0]; }
}
 
int main( )
{
    using T = double;
    using P = T*;

    // Size of vectors
    int n = 256;

    // Number of threads in each thread block
    int blockSize = 16;

    // Size, in bytes, of input vector
    size_t szi = n*sizeof(T);

    // Size, in bytes, of output vector
    size_t szo = n/blockSize*sizeof(T);
 
    // Host vectors
    std::vector<T> h_src(n);
    std::vector<T> h_dst(n/blockSize);
   
    // Device vectors
    P d_src, d_dst;

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_src, szi);
    cudaMalloc(&d_dst, szo);
 
    // Initialize vectors on host
    for(size_t i = 0; i < n; i++ )
    {
        h_src[i] = i*0.001;
    }

    for(size_t i = 0; i < h_dst.size(); i++ )
    {
        h_dst[i] = 0;
    }

    // Copy host vectors to device
    cudaMemcpy( d_src, h_src.data(), szi, cudaMemcpyHostToDevice);
    cudaMemcpy( d_dst, h_dst.data(), szo, cudaMemcpyHostToDevice);
 
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/2/blockSize);
 
    // Execute the kernel
    auto sum = [] __host__ __device__ (T const& x, T const& y){ return x + y; };
    
    reduce<<<gridSize, blockSize, blockSize*sizeof(T)>>>(sum, d_src, d_dst);
 
    // Copy array back to host
    cudaMemcpy( h_dst.data(), d_dst, szo, cudaMemcpyDeviceToHost );
 
    T res = 0.0;
    for(size_t i=0; i<h_dst.size(); i++)
    {
        res = sum(res, h_dst[i]);
    }

    std::cout << "result = " << res << "\n";
    // Release device memory
    cudaFree(d_src);
    cudaFree(d_dst);
 
    return 0;
}
