#include <CUDA-C++-API.hpp>


int main(void)
{
  try
  {
  // Init computation
  const std::size_t chainlength = std::size_t(std::pow(2u, 20u)); // 1M, cast denotes floating-to-integral conversion,
                                                                  //     promises no data is lost, silences compiler warning
  std::valarray<float> vec_x( chainlength ),
                       vec_y( chainlength );
  float a = 2.0;

  // Fill arrays with random values between 0 and 100
  auto prng = [engine = std::default_random_engine{},
               distribution = std::uniform_real_distribution<float>{ -100.0f, 100.0f }]() mutable { return distribution(engine); };

  std::generate_n(std::begin(vec_x), chainlength, prng);
  std::generate_n(std::begin(vec_y), chainlength, prng);

  float *d_x = nullptr, *d_y = nullptr;

  cudaMalloc(&d_x, chainlength * sizeof(float));
  cudaMalloc(&d_y, chainlength * sizeof(float));

  // Explicit (non-blocking) dispatch of data before launch
  cudaMemcpy(d_x, &vec_x[0], chainlength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, &vec_x[0], chainlength * sizeof(float), cudaMemcpyHostToDevice);

  // Compute SAXPY on device
  cuda::launch<<<(chainlength+255)/256, 256>>>([=] __device__ ()
  {
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      
      if (i < chainlength) d_y[i] = a * d_x[i] + d_y[i];
  });

  // Compute validation set on host
  std::valarray<float> ref = a * vec_x + vec_y;

  cudaMemcpy(&vec_y[0], d_y, chainlength * sizeof(float), cudaMemcpyDeviceToHost);

  // Validate (compute saxpy on host and match results)
  auto markers = std::mismatch(std::cbegin(vec_y), std::cend(vec_y),
                               std::cbegin(ref), std::cend(ref));

  if (markers.first != std::cend(vec_y) || markers.second != std::cend(ref)) throw std::runtime_error{ "Validation failed." };

  cudaFree(d_x);
  cudaFree(d_y);
  }
  catch (std::exception error) // If STL/CRT error occurs
  {
    std::cerr << error.what() << std::endl;

    std::exit( EXIT_FAILURE );
  }

  return EXIT_SUCCESS;
}
