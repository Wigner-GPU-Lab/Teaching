#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

struct v3{ float x, y, z; };
struct v4{ float x, y, z, m; };

__host__ __device__ float sq(float x){ return x*x; }
__host__ __device__ float cube(float x){ return x*x*x; }

void cpu_nbody_opt(std::vector<v3>& F, std::vector<v4> const& P, float G, float eps) 
{
    int N = (int)P.size();
    for(int i=0; i<N; ++i)
    {
        v3 sum = {0.0f, 0.0f, 0.0f};
        float x = P[i].x;
        float y = P[i].y;
        float z = P[i].z;
        for(int j=0; j<N; ++j)
        {
            if(i == j){ continue; }
            float dx = P[j].x - x;
            float dy = P[j].y - y;
            float dz = P[j].z - z;
            float rec = P[j].m / (/*eps + */cube( sqrt(sq(dx) + sq(dy) + sq(dz)) ));
            sum.x += dx * rec;
            sum.y += dy * rec;
            sum.z += dz * rec;
        }
        float gm = -G * P[i].m;
        F[i] =  v3{gm * sum.x, gm * sum.y, gm * sum.z};
    }
}

__global__ void gpu_nbody_opt(float3* F, float4* P, float G, float eps) 
{
    unsigned int N = blockDim.x*gridDim.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float3 sum1 = {0.0f, 0.0f, 0.0f};
    float3 sum2 = {0.0f, 0.0f, 0.0f};
    float4 Pi1 = P[i];
    float4 Pi2 = P[i+N];
    float x1 = Pi1.x; float y1 = Pi1.y; float z1 = Pi1.z;
    float x2 = Pi2.x; float y2 = Pi2.y; float z2 = Pi2.z;
    for(int j=0; j<N*2; ++j)
    {
        if(i == j){ continue; }
        float4 Pj = P[j];
        float dx1 = Pj.x - x1, dy1 = Pj.y - y1, dz1 = Pj.z - z1;
        float dx2 = Pj.x - x2, dy2 = Pj.y - y2, dz2 = Pj.z - z2;

        float rec1 = Pj.w * cube( rsqrtf( sq(dx1) + sq(dy1) + sq(dz1) ) );
        float rec2 = Pj.w * cube( rsqrtf( sq(dx2) + sq(dy2) + sq(dz2) ) );

        sum1.x += dx1 * rec1; sum1.y += dy1 * rec1; sum1.z += dz1 * rec1;
        sum2.x += dx2 * rec2; sum2.y += dy2 * rec2; sum2.z += dz2 * rec2;
    }
    float gm1 = -G * Pi1.w;
    float gm2 = -G * Pi2.w;
    F[i]   = float3{gm1*sum1.x, gm1*sum1.y, gm1*sum1.z};
    F[i+N] = float3{gm2*sum2.x, gm2*sum2.y, gm2*sum2.z};
}

int main()
{
    const int N = 8192*2;
    const int block_sz = 64;
    const int n_blocks = N / block_sz;
    const float G = 1e-2;
    const float eps = 1e-5;

    std::vector<v4> Points(N);
    std::vector<v3> Forces1(N), Forces2(N);

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};

    auto gen = [&dist, &mersenne_engine](){ return v4{dist(mersenne_engine), dist(mersenne_engine), dist(mersenne_engine), dist(mersenne_engine)+0.3f}; };
    generate(Points.begin(), Points.end(), gen);
    std::fill(Forces1.begin(), Forces1.end(), v3{0.0f, 0.0f, 0.0f});
    std::fill(Forces2.begin(), Forces2.end(), v3{0.0f, 0.0f, 0.0f});
	
	float* pP = nullptr;
    float* pF = nullptr;

    cudaEvent_t evt[2];
    for(auto& e : evt){ cudaEventCreate(&e); }

	cudaError_t err = cudaSuccess;
	err = cudaMalloc( (void**)&pP, N*sizeof(v4) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&pF, N*sizeof(v3) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( pP, Points.data(), N*sizeof(v4), cudaMemcpyHostToDevice );
    if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    {
        dim3 dimGrid( n_blocks/2 );
        dim3 dimBlock( block_sz );
        cudaEventRecord(evt[0]);
        gpu_nbody_opt<<<dimGrid, dimBlock>>>((float3*)pF, (float4*)pP, G, eps);
        err = cudaGetLastError();
	    if (err != cudaSuccess){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
        cudaEventRecord(evt[1]);
    }

	err = cudaMemcpy( Forces2.data(), pF, N*sizeof(v3), cudaMemcpyDeviceToHost );
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
    cpu_nbody_opt(Forces1, Points, G, eps);
    auto t1 = std::chrono::high_resolution_clock::now();

    bool mismatch = false;
    const float max_err = 5e-4f;
    for(int i=0; i<N; ++i)
	{
        v3 f1 = Forces1[i];
        v3 f2 = Forces2[i];
        float length_error = std::sqrt((f1.x-f2.x)*(f1.x-f2.x) + (f1.y-f2.y)*(f1.y-f2.y) + (f1.z-f2.z)*(f1.z-f2.z));
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
