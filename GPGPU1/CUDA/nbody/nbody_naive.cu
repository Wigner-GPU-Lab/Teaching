#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

struct Force{ float x, y, z; };
struct Particle{ float x, y, z, m; };

void cpu_nbody_naive(std::vector<Force>& F, std::vector<Particle> const& P, float G, float eps) 
{
    int N = (int)P.size();
    for(int i=0; i<N; ++i)
    {
        Force sum = {0.0f, 0.0f, 0.0f};
        for(int j=0; j<N; ++j)
        {
            sum.x += -G * P[i].m * P[j].m * (P[j].x - P[i].x) /
                (eps + std::pow( std::sqrt( std::pow(P[j].x - P[i].x, 2.0f) + std::pow(P[j].y - P[i].y, 2.0f) + std::pow(P[j].z - P[i].z, 2.0f) ), 3.0));
            sum.y += -G * P[i].m * P[j].m * (P[j].y - P[i].y) /
                (eps + std::pow( std::sqrt( std::pow(P[j].x - P[i].x, 2.0f) + std::pow(P[j].y - P[i].y, 2.0f) + std::pow(P[j].z - P[i].z, 2.0f) ), 3.0));
            sum.z += -G * P[i].m * P[j].m * (P[j].z - P[i].z) /
                (eps + std::pow( std::sqrt( std::pow(P[j].x - P[i].x, 2.0f) + std::pow(P[j].y - P[i].y, 2.0f) + std::pow(P[j].z - P[i].z, 2.0f) ), 3.0));
        }
        F[i] = sum;
    }
}

__global__ void gpu_nbody_naive(float3* F, float4* P, float G, float eps) 
{
    unsigned int N = blockDim.x*gridDim.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float3 sum = {0.0f, 0.0f, 0.0f};
    for(int j=0; j<N; ++j)
    {
        sum.x += -G * P[i].w * P[j].w * (P[j].x - P[i].x) /
            (eps + std::pow( std::sqrt( std::pow(P[j].x - P[i].x, 2.0f) + std::pow(P[j].y - P[i].y, 2.0f) + std::pow(P[j].z - P[i].z, 2.0f) ), 3.0f));
        sum.y += -G * P[i].w * P[j].w * (P[j].y - P[i].y) /
            (eps + std::pow( std::sqrt( std::pow(P[j].x - P[i].x, 2.0f) + std::pow(P[j].y - P[i].y, 2.0f) + std::pow(P[j].z - P[i].z, 2.0f) ), 3.0f));
        sum.z += -G * P[i].w * P[j].w * (P[j].z - P[i].z) /
            (eps + std::pow( std::sqrt( std::pow(P[j].x - P[i].x, 2.0f) + std::pow(P[j].y - P[i].y, 2.0f) + std::pow(P[j].z - P[i].z, 2.0f) ), 3.0f));
    }
    F[i] = sum;
}

int main()
{
    const int N = 8192*2;
    const int block_sz = 64;
    const int n_blocks = N / block_sz;
    const float G = 1e-2;
    const float eps = 1e-5;

    std::vector<Particle> Points(N);
    std::vector<Force> Forces1(N), Forces2(N);

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};

    auto gen = [&dist, &mersenne_engine](){ return Particle{dist(mersenne_engine), dist(mersenne_engine), dist(mersenne_engine), dist(mersenne_engine)+0.3f}; };
    generate(Points.begin(), Points.end(), gen);
    std::fill(Forces1.begin(), Forces1.end(), Force{0.0f, 0.0f, 0.0f});
    std::fill(Forces2.begin(), Forces2.end(), Force{0.0f, 0.0f, 0.0f});
	
	float* pP = nullptr;
    float* pF = nullptr;

    cudaEvent_t evt[2];
    for(auto& e : evt){ cudaEventCreate(&e); }

	cudaError_t err = cudaSuccess;
	err = cudaMalloc( (void**)&pP, N*sizeof(Particle) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&pF, N*sizeof(Force) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( pP, Points.data(), N*sizeof(Particle), cudaMemcpyHostToDevice );
    if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    {
        dim3 dimGrid( n_blocks );
        dim3 dimBlock( block_sz );
        cudaEventRecord(evt[0]);
        gpu_nbody_naive<<<dimGrid, dimBlock>>>((float3*)pF, (float4*)pP, G, eps);
        err = cudaGetLastError();
	    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
        cudaEventRecord(evt[1]);
    }

	err = cudaMemcpy( Forces2.data(), pF, N*sizeof(Force), cudaMemcpyDeviceToHost );
	if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( pP );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( pF );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    cudaEventSynchronize(evt[1]);
    float dt = 0.0f;//milliseconds
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    for(auto& e : evt){ cudaEventDestroy(e); }

    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_nbody_naive(Forces1, Points, G, eps);
    auto t1 = std::chrono::high_resolution_clock::now();

    bool mismatch = false;
    const float max_err = 5e-4f;
    auto sq = [](float x){ return x*x; };
    for(int i=0; i<N; ++i)
	{
        Force f1 = Forces1[i];
        Force f2 = Forces2[i];
        float length_error = std::sqrt( sq(f1.x-f2.x) + sq(f1.y-f2.y) + sq(f1.z-f2.z) );
        if( length_error > max_err )
        {
            mismatch = true;
            std::cout << "[" << i << "] : " << f1.x << "   " << f2.x << ",   " << f1.y << "   " << f2.y << ",   " << f1.z << "   " << f2.z << " difference length: " << length_error << "\n";
        }
    }

    if( !mismatch )
	{
		std::cout << "Success.\n";
	}
	else
	{
		std::cout << "Mismatch in CPU and GPU result.\n";
	}
    
    std::cout << "CPU Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "GPU Computation took: " << dt << " ms.\n";
	return 0;
}
