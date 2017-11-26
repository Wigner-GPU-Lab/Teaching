#include <vector>
#include <iostream>
#include <fstream>

template<typename F, typename T, typename R>
__global__ void sliding_map_3_impl(F f, T* src, R* dst)
{
    int ix = blockIdx.x*blockDim.x+threadIdx.x;
    int iy = blockIdx.y*blockDim.y+threadIdx.y;
    int w = blockDim.y * gridDim.y + 2;
    int o = iy * w + ix;

    dst[iy * (w-2) + ix] = f(src[o],     src[o+1],     src[o+2],
                             src[o+w],   src[o+w+1],   src[o+w+2],
                             src[o+2*w], src[o+2*w+1], src[o+2*w+2]);
}

/*template<typename F, typename T, typename R>
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
}*/

template<typename T, typename F, typename R = T/*typename std::result_of<F(T, T, T, T, T, T, T, T, T)>::type*/>
auto sliding_map_3(F f, std::vector<T>const& src, int nx, int ny)
{
    size_t n = src.size();

    dim3 dimBlock(4, 4);
    dim3 dimGrid( (nx-2)/4, (ny-2)/4 );

    std::vector<R> res( (nx-2)*(ny-2) );

    T* d_src;
    R* d_res;

    cudaMalloc(&d_src, n*sizeof(T));
    cudaMalloc(&d_res, res.size()*sizeof(R));

    cudaMemcpy( d_src, src.data(), n*sizeof(T), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sliding_map_3_impl<<<dimGrid, dimBlock>>>(f, d_src, d_res);
    cudaEventRecord(stop);    
    
    cudaEventSynchronize(stop);    
    float cuda_time = 0.0f;//msec
    cudaEventElapsedTime(&cuda_time, start, stop);
    std::cout << "Elapsed time is: " << cuda_time << " msec\n";
    cudaDeviceSynchronize();
    cudaMemcpy( res.data(), d_res, res.size()*sizeof(R), cudaMemcpyDeviceToHost );

    cudaFree(d_src);
    cudaFree(d_res);

    return res;
}

template<typename T>
T sq(T const& x){ return x*x; }
 
int main( )
{
    using T = double;
   
    int nx = 512+2;
    int ny = 512+2;
    int n = nx*ny;

    std::vector<T> v(n);
   
    double x0 = -6.0;
    double x1 = +6.0;
    double y0 = -6.0;
    double y1 = +6.0;
    auto dx = (x1-x0) / (nx-1.0);
    auto dy = (y1-y0) / (ny-1.0);
    
    for(int i = 0; i < ny; i++ )
    {
       auto y = (i / (ny-1.0)) * (y1-y0) + y0;
       for(int j = 0; j < nx; j++ )
       {
          auto x = (j / (nx-1.0)) * (x1-x0) + x0;
          v[i*ny+j] = exp(-(sq(x)+sq(y)));//sq(a-x) + b*sq(y - sq(x));
       }
    }
 
    auto diff = [=] __host__ __device__ (T a00, T a01, T a02,
                                         T a10, T a11, T a12,
                                         T a20, T a21, T a22)->T{ return (a00-a20 + a22 - a02)/dx/dy/2.0; };

    auto laplacian = [=] __host__ __device__ (T a00, T a01, T a02,
                                              T a10, T a11, T a12,
                                              T a20, T a21, T a22)->T{ return (a00+a02+a20+a22 + 4.0*(a10+a01+a12+a21)-20*a11)/dx/dy/6.0; };
    
    auto dfdx   = sliding_map_3(laplacian, v, nx, ny);
 
    {
        std::ofstream file("2Dstencil.txt");
	for(int i = 1; i < ny-1; i++ )
        {
          auto y = (i / (ny-1.0)) * (y1-y0) + y0;
          for(int j = 1; j < nx-1; j++ )
          {
            auto x = (j / (nx-1.0)) * (x1-x0) + x0;
            file << x << "   " << y << "   " <<  v[i*ny+j] << "   " << dfdx[(i-1)*(ny-2)+(j-1)] << "\n";
          }
          file << "\n";
        }
    }

    return 0;
}
