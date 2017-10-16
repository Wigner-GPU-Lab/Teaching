#include <CUDA-C++-API.hpp>


int main(void)
{
  int N = 1<<20;
  std::vector<float> x(N, 1.0f), y(N, 2.0f);
  float *d_x = nullptr, *d_y = nullptr, a = 2.0;

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  auto saxpy = [=] __device__ ()
  {
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      
      if (i < N) d_y[i] = a * d_x[i] + d_y[i];
  };

  // Perform SAXPY on 1M elements
  cuda::launch_kernel<<<(N+255)/256, 256>>>(saxpy);

  cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float res = a * 1.0f + 2.0f;
  float maxError = *std::max_element(y.cbegin(),
                                     y.cend(),
                                     [=](const float& lhs, const float& rhs)
  {
    return std::abs(lhs - res) < std::abs(rhs - res);
  });

  std::cout.precision(16);
  std::cout << "Max error: " << std::abs(maxError - res) << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
}
