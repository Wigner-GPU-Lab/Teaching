#include <vector>
#include <iostream>
#include <fstream>

template<typename F, typename T, typename R>
__global__ void sliding_map_3_impl(F f, T left, T right, T* src, R* dst)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int imax = blockDim.x * gridDim.x - 1;
    if     (id == 0   ){ dst[id] = f(left,        src[0   ], src[1   ]); }
    else if(id == imax){ dst[id] = f(src[imax-1], src[imax], right    ); }
    else               { dst[id] = f(src[id  -1], src[id  ], src[id+1]); }
}

template<typename F, typename T, typename R>
__global__ void sliding_map_3_impl2(F f, T left, T right, T* src, R* dst)
{
    extern __shared__ T tmp[];

    int i    = blockIdx.x*blockDim.x+threadIdx.x;
    int imax = blockDim.x * gridDim.x - 1;
    int t    = threadIdx.x;
    int tmax = blockDim.x - 1;
    if     (t == 0   ){ if(i == 0   ){ tmp[t+0] = left;  }else{ tmp[t+0] = src[i-1]; } }
    else if(t == tmax){ if(i == imax){ tmp[t+2] = right; }else{ tmp[t+2] = src[i+1]; } }
    tmp[t+1] = src[i];
    
    __syncthreads();
    
    dst[i] = f(tmp[t], tmp[t+1], tmp[t+2]);
}

template<typename T, typename F, typename R = typename std::result_of<F(T, T, T)>::type>
auto sliding_map_3(F f, T const& left, T const& right, std::vector<T>const& src)
{
    //using R = decltype(f(left, left, right));
    size_t n = src.size();
    static const size_t blockSize = 32;
           const size_t gridSize  = (size_t)ceil((float)n/blockSize);

    std::vector<R> res(n);

    T* d_src;
    R* d_res;

    cudaMalloc(&d_src, n*sizeof(T));
    cudaMalloc(&d_res, n*sizeof(R));

    cudaMemcpy( d_src, src.data(), n*sizeof(T), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sliding_map_3_impl2<<<gridSize, blockSize, (blockSize+2)*sizeof(T)>>>(f, left, right, d_src, d_res);
    cudaEventRecord(stop);    
    
    cudaEventSynchronize(stop);    
    float cuda_time = 0.0f;//msec
    cudaEventElapsedTime(&cuda_time, start, stop);
    std::cout << "Elapsed time is: " << cuda_time << " msec\n";
    cudaDeviceSynchronize();
    cudaMemcpy( res.data(), d_res, n*sizeof(R), cudaMemcpyDeviceToHost );

    cudaFree(d_src);
    cudaFree(d_res);

    return res;
}
 
int main( )
{
    using T = double;
   
    int n = 4096*256*2;

    std::vector<T> v(n);
   
    double x0 = -3.0;
    double x1 = +3.0;
    auto dx = (x1-x0) / (n-1.0);
    for(int i = 0; i < n; i++ )
    {
        auto x = (i / (n-1.0)) * (x1-x0) + x0;
        v[i] = exp(-x*x);
    }
 
    auto diff1 = [=] __host__ __device__ (T y0, T y1, T y2)->T{ return 0.5*(y2-y0)/dx; };
    auto diff2 = [=] __host__ __device__ (T y0, T y1, T y2)->T{ return (y0-2.0*y1+y2)/(dx*dx); };
    
    auto dfdx   = sliding_map_3(diff1, 0.0, 0.0, v);
    auto d2fdx2 = sliding_map_3(diff2, 0.0, 0.0, v);
 
    {
        std::ofstream file("stencil.txt");
	for(int i=0; i<n; i++)
 	{
           file << (i / (n-1.0)) * (x1-x0) + x0 << "   " << v[i] << "   " << dfdx[i] << "   " << d2fdx2[i]<< "\n";
        }
    }

    return 0;
}
