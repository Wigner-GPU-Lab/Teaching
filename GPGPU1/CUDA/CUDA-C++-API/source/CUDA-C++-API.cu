#include <iostream>

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

  __global__ auto saxpy = [=, a = 2.0f]()
  {
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      return d_y[i] = a * d_x[i] + d_y[i];
  };

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>();

  cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  auto maxError = std::

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));

  std::cout << "Max error: "<< maxError << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
}
