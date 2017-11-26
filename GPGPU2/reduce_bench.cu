#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>

template<typename T>
void gen_rand_vct(std::vector<T>& v, size_t n)
{
	std::minstd_rand engine(42);
	std::uniform_real_distribution<T> dist((T)0.0, (T)10.0);

	auto gen = std::bind(dist, engine);
	v.resize(n);
	//for(auto& x : v){ x = (T)1; }
	std::generate(begin(v), end(v), gen);
}

template<typename F, typename T>
__global__ void reduce_kernel(F f, T* src, T* dst)
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

template<typename Fr, typename T>
std::pair<T, long long> gpu_reduce(Fr fr, std::vector<T> const& v)
{
  using P = T*;
  size_t n = v.size();

  //block size, assumes input > n
  size_t local_count = 256;

  // Size, in bytes, of input vector
  size_t szi = n*sizeof(T);
 
  std::vector<T> tmp( (size_t)std::ceil(n/2.0/local_count) );
  size_t szo = tmp.size() * sizeof(T);

  T res = (T)0;

  // Device vectors
  P d_src, d_dst;

  // Allocate memory for each vector on GPU
  cudaMalloc(&d_src, szi);
  cudaMalloc(&d_dst, szo);

  // Copy host vector to device
  cudaMemcpy( d_src, v.data(), szi, cudaMemcpyHostToDevice);

  int gridSize = (int)ceil((float)n/2/local_count);
  
  auto t0 = std::chrono::high_resolution_clock::now();

  // Execute the kernel
  reduce_kernel<<<gridSize, local_count, local_count*sizeof(T)>>>(fr, d_src, d_dst);
 
  // Copy array back to host
  cudaMemcpy( tmp.data(), d_dst, szo, cudaMemcpyDeviceToHost );

  if(tmp.size() == 1){ res = tmp[0]; }
  else{ res = std::accumulate( tmp.cbegin()+1, tmp.cend(), tmp[0], fr ); }

  auto t1 = std::chrono::high_resolution_clock::now();

  cudaFree(d_src);
  cudaFree(d_dst);

  return std::make_pair(res, std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
}

int main()
{
    using T = double;
	auto fr = []__host__ __device__(T const& x, T const& y){ return x+y; };
        cudaSetDevice(2);
	std::ofstream file("reduce2.txt");
	for(size_t i=6; i<=28; ++i)
	{
		std::vector<T> v;

		std::cout << i << "\n";

		gen_rand_vct(v, 1 << i);

		auto r_gpu = gpu_reduce(fr, v);
		auto t1 = std::chrono::high_resolution_clock::now();
			
		auto r_cpu = std::accumulate( v.cbegin()+1, v.cend(), v[0], fr );
		auto t2 = std::chrono::high_resolution_clock::now();

		auto dt_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
		auto dt_gpu = r_gpu.second;

		std::cout << "CPU Reduce took: " << dt_cpu << " microseconds." << std::endl;
		std::cout << "GPU Reduce took: " << dt_gpu << " microseconds." << std::endl;
		file <<  i << " " << dt_cpu << " " << dt_gpu << "\n";

		printf("CPU result = %16.16f\n", r_cpu);
		printf("GPU result = %16.16f\n", r_gpu.first);
	}
	return 0;
}
