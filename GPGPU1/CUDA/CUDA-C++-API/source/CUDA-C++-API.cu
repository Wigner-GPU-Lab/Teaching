#include <CUDA-C++-API.hpp>


int main(void)
{
  int N = 1<<20;
  std::vector<float> x(N, 1.0f), y(N, 2.0f);
  float *d_x = nullptr, *d_y = nullptr;

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  auto saxpy = [=, a = 2.0f] __device__ ()
  {
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      d_y[i] = 6;// a * d_x[i] + d_y[i];
  };

  // Perform SAXPY on 1M elements
  cuda::launch_kernel<<<(N+255)/256, 256>>>(saxpy);

  cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = *std::max_element(y.cbegin(), y.cend(),
                                    [res = 4.0](const float& lhs,
                                                const float& rhs)
  {
    return std::abs(lhs - res) < std::abs(rhs - res);
  });
  std::cout << "Max error: "<< maxError << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
}
