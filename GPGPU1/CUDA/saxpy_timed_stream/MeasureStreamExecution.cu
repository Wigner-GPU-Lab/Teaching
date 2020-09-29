#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

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
    size_t sz = 1 << 24;
    std::cout << "Data size is: " << sz << "\n";
    const float A = 2.0f;

	std::vector<float> X(sz);
	std::vector<float> Y(sz);
    std::vector<float> R(sz);
    
    std::generate(X.begin(), X.end(), [d=0.f, dd=+1.0f/sz]()mutable{ d += dd; return d; });
    std::generate(Y.begin(), Y.end(), [d=1.f, dd=-1.0f/sz]()mutable{ d += dd; return d; });
	
	float* devX = nullptr;
	float* devY = nullptr;

    cudaError_t err = cudaSuccess;

    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if( err != cudaSuccess ){ std::cout << "Error creating CUDA stream: " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&devX, sz*sizeof(float) );
	if( err != cudaSuccess ){ std::cout << "Error allocating CUDA memory (X): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	err = cudaMalloc( (void**)&devY, sz*sizeof(float) );
	if( err != cudaSuccess ){ std::cout << "Error allocating CUDA memory (Y): " << cudaGetErrorString(err) << "\n"; return -1; }
	
	//err = cudaMemcpy( devX, X.data(), sz*sizeof(float), cudaMemcpyHostToDevice );
    err = cudaMemcpyAsync(devX, X.data(), sz*sizeof(float), cudaMemcpyHostToDevice, stream);
    if( err != cudaSuccess ){ std::cout << "Error copying memory to device (X): " << cudaGetErrorString(err) << "\n"; return -1; }
    
    //err = cudaMemcpy( devY, Y.data(), sz*sizeof(float), cudaMemcpyHostToDevice );
    err = cudaMemcpyAsync(devY, Y.data(), sz*sizeof(float), cudaMemcpyHostToDevice, stream);
    if( err != cudaSuccess ){ std::cout << "Error copying memory to device (Y): " << cudaGetErrorString(err) << "\n"; return -1; }
    
    cudaEvent_t evt[3];
    for(auto& e : evt)
    {
        err = cudaEventCreate(&e);
        if( err != cudaSuccess ){ std::cout << "Error creating event: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

	dim3 dimGrid( sz/512 );
    dim3 dimBlock( 512 );
    
    err = cudaEventRecord(evt[0], stream);
    if( err != cudaSuccess ){ std::cout << "Error recording event (0): " << cudaGetErrorString(err) << "\n"; return -1; }
    
    saxpy<<<dimGrid, dimBlock, 0, stream>>>((int)sz, A, devX, devY);
    
    err = cudaGetLastError();
	if (err != cudaSuccess ){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    err = cudaEventRecord(evt[1], stream);
    if( err != cudaSuccess ){ std::cout << "Error recording event (1): " << cudaGetErrorString(err) << "\n"; return -1; }

	//err = cudaMemcpy( R.data(), devY, sz*sizeof(float), cudaMemcpyDeviceToHost );
    err = cudaMemcpyAsync( R.data(), devY, sz*sizeof(float), cudaMemcpyDeviceToHost, stream );
	if( err != cudaSuccess ){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaEventRecord(evt[2], stream);
    if( err != cudaSuccess ){ std::cout << "Error recording event (2): " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaEventSynchronize(evt[2]);
    if( err != cudaSuccess ){ std::cout << "Error during synchronize with event: " << cudaGetErrorString(err) << "\n"; return -1; }

    float dt1 = 0.0f, dt2 = 0.0f;//milliseconds
    err = cudaEventElapsedTime(&dt1, evt[0], evt[1]);
    if( err != cudaSuccess ){ std::cout << "Error getting event 0-1 elapsed time: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaEventElapsedTime(&dt2, evt[1], evt[2]);
    if( err != cudaSuccess ){ std::cout << "Error getting event 1-2 elapsed time: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( devX );
    if( err != cudaSuccess ){ std::cout << "Error freeing allocation (X): " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( devY );
    if( err != cudaSuccess ){ std::cout << "Error freeing allocation (Y): " << cudaGetErrorString(err) << "\n"; return -1; }

    for(auto& e : evt)
    {
        err = cudaEventDestroy(e);
        if( err != cudaSuccess ){ std::cout << "Error destroying event: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    err = cudaStreamDestroy(stream);
    if( err != cudaSuccess ){ std::cout << "Error destroying CUDA stream: " << cudaGetErrorString(err) << "\n"; return -1; }

    // skip printing:
	/*for( auto r : R )
	{
		std::cout << r << "\n";
	}*/

    auto t0 = std::chrono::high_resolution_clock::now();
	std::transform(X.begin(), X.end(), Y.begin(), Y.begin(), [a = A](float x, float y){ return a*x + y; });
    auto t1 = std::chrono::high_resolution_clock::now();
    
	if(std::equal(R.begin(), R.end(), Y.begin()))
	{
		std::cout << "Success\n";
	}
    else{ std::cout << "Mismatch between CPU and GPU results.\n"; }
    
    std::cout << "CPU computation took:         " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "GPU computation took:         " << dt1 << " ms.\n";
    std::cout << "GPU device-to-host copy took: " << dt2 << " ms.\n";

	return 0;
}
