#include <iostream>   // std::cout
#include <algorithm>  // std::max_element
#include <vector>     // std::vector
#include <cmath>      // std::abs


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  std::vector<float> x(N, 1.0f), y(N, 2.0f);
  float *d_x, *d_y;

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  auto saxpy = [=, a = 2.0f]() __global__
  {
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      return d_y[i] = a * d_x[i] + d_y[i];
  };

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>();

  cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = std::max_element(y.cbegin(), y.cend(),
                                    [res = 4.0](const float& lhs,
                                                const float& rhs)
  {
    return std::abs(lhs - res) < std::abs(rhs - res;)
  };)
  std::cout << "Max error: "<< maxError << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
}
