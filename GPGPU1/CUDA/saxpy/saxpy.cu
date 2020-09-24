#include <vector>
#include <algorithm>
#include <iostream>

__global__ void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        y[i] = a*x[i] + y[i];
    }
}

int main()
{
    const float A = 100.0f;
	std::vector<float> X{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
	std::vector<float> Y{0.f, 2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f, 16.f, 18.f};
	std::vector<float> R(X.size());

	size_t sz = X.size();
	float* devX = nullptr;
	float* devY = nullptr;

	cudaError_t err = cudaSuccess;
	err = cudaMalloc( (void**)&devX, sz*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory (X): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&devY, sz*sizeof(float) );
	if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory (Y): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMemcpy( devX, X.data(), sz*sizeof(float), cudaMemcpyHostToDevice );
	if( err != cudaSuccess){ std::cout << "Error copying memory to device (X): " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaMemcpy( devY, Y.data(), sz*sizeof(float), cudaMemcpyHostToDevice );
	if( err != cudaSuccess){ std::cout << "Error copying memory to device (Y): " << cudaGetErrorString(err) << "\n"; return -1; }
    
	dim3 dimGrid( 1 );
	dim3 dimBlock( sz );
	saxpy<<<dimGrid, dimBlock>>>((int)sz, A, devX, devY);

	err = cudaGetLastError();
	if (err != cudaSuccess){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaMemcpy( R.data(), devY, sz*sizeof(float), cudaMemcpyDeviceToHost );
	if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( devX );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation (X): " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFree( devY );
	if( err != cudaSuccess){ std::cout << "Error freeing allocation (Y): " << cudaGetErrorString(err) << "\n"; return -1; }

	for( auto r : R )
	{
		std::cout << r << "\n";
	}

	std::transform(X.begin(), X.end(), Y.begin(), Y.begin(), [a = A](float x, float y){ return a*x + y; });

	if(std::equal(R.begin(), R.end(), Y.begin()))
	{
		std::cout << "Success\n";
	}
	else{ std::cout << "Mismatch between CPU and GPU results.\n"; }
	
	return 0;
}
