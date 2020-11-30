#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

struct color{ unsigned char r, g, b, a; };
struct three_histograms
{
    std::array<unsigned int, 256> rh, gh, bh;
    void make_null()
    {
        for(int i=0; i<256; ++i)
        {
            rh[i] = 0; gh[i] = 0; bh[i] = 0;
        }
    }

    void fromLinearMemory( std::vector<unsigned int>& input )
    {
        for(int i=0; i<256; ++i)
        {
            rh[i] = input[0*256+i];
            gh[i] = input[1*256+i];
            bh[i] = input[2*256+i];
        }
    }
};

void cpu_histo( three_histograms& output, color* const& input, int W, int H )
{
    for(int y=0; y<H; ++y)
    {
        for(int x=0; x<W; ++x)
        {
            color c = input[y*W+x];
            output.rh[c.r] += 1;
            output.gh[c.g] += 1;
            output.bh[c.b] += 1;
        }
    }
}

__global__ void gpu_histo_global_atomics( unsigned int* output, uchar4* input, int W )
{
    // linear block index within 2D grid
    int B = blockIdx.x + blockIdx.y * gridDim.x;

    //Output index start for this block's histogram:
    int I = B*(3*256);
    unsigned int* H = output + I;

    // process pixel blocks horizontally
    // updates our block's partial histogram in global memory
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int x = threadIdx.x; x < W; x += blockDim.x)
    {
        uchar4 pixels = input[y * W + x];
        atomicAdd(&H[0 * 256 + pixels.x], 1);
        atomicAdd(&H[1 * 256 + pixels.y], 1);
        atomicAdd(&H[2 * 256 + pixels.z], 1);
    }
}

__global__ void gpu_histo_shared_atomics( unsigned int* output, uchar4* input, int W )
{
    //histograms are in shared memory:
    __shared__ unsigned int histo[3 * 256];

    //Number of threads in the block:
    int Nthreads = blockDim.x * blockDim.y;
    //Linear thread idx:
    int LinID = threadIdx.x + threadIdx.y * blockDim.x;
    //zero histogram:
    for (int i = LinID; i < 3*256; i += Nthreads){ histo[i] = 0; }
    __syncthreads();

    // linear block index within 2D grid
    int B = blockIdx.x + blockIdx.y * gridDim.x;

    // process pixel blocks horizontally
    // updates the partial histogram in shared memory
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int x = threadIdx.x; x < W; x += blockDim.x)
    {
        uchar4 pixels = input[y * W + x];
        atomicAdd(&histo[0 * 256 + pixels.x], 1);
        atomicAdd(&histo[1 * 256 + pixels.y], 1);
        atomicAdd(&histo[2 * 256 + pixels.z], 1);
    }

    __syncthreads();

    //Output index start for this block's histogram:
    int I = B*(3*256);
    unsigned int* H = output + I;

    //Copy shared memory histograms to globl memory:
    for (int i = LinID; i < 256; i += Nthreads)
    {
        H[0*256 + i] = histo[0*256 + i];
        H[1*256 + i] = histo[1*256 + i];
        H[2*256 + i] = histo[2*256 + i];
    }
}

__global__ void gpu_histo_accumulate(const unsigned int* in, int nBlocks, unsigned int* out)
{
    //each thread sums one shade of the r, g, b histograms
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 3 * 256)
    {
        unsigned int sum = 0;
        for(int j = 0; j < nBlocks; j++)
        {
            sum += in[i + (3*256) * j];
        }            
        out[i] = sum;
    }
}

int main()
{
    static const std::string input_filename   = "NZ.jpg";
    static const std::string output_filename1 = "cpu_out.jpg";
    static const std::string output_filename2 = "gpu_out1.jpg";
    static const std::string output_filename3 = "gpu_out2.jpg";

    static const int block_size = 16;
    //int nBlocksW = 0; //number of blocks horizontally, not used now
    int nBlocksH = 0; //number of blocks vertically
    
    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components

    color* data0 = reinterpret_cast<color*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        //nBlocksW = w / block_size; //not used now
        nBlocksH = h / block_size;
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }
    
    three_histograms cpu;  cpu.make_null();
    three_histograms gpu1; gpu1.make_null();
    three_histograms gpu2; gpu2.make_null();
   
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_histo(cpu, data0, w, h);
    auto t1 = std::chrono::high_resolution_clock::now();

    //GPU version using global atomics:
    float dt1 = 0.0f;
    {
        unsigned char* pInput    = nullptr;
        unsigned int*  pPartials = nullptr;
        unsigned int*  pOutput   = nullptr;

        cudaError_t err = cudaSuccess;
        err = cudaMalloc( (void**)&pInput, w*h*sizeof(color) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMalloc( (void**)&pPartials, nBlocksH*3*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemset(pPartials, 0, nBlocksH*3*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        err = cudaMalloc( (void**)&pOutput, 3*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemcpy( pInput, data0, w*h*sizeof(color), cudaMemcpyHostToDevice );
        if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

        cudaEvent_t evt[4];
        for(auto& e : evt){ cudaEventCreate(&e); }
        
        //First kernel of global histograms:
        {
            dim3 dimGrid( 1, nBlocksH );
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[0]);
            gpu_histo_global_atomics<<<dimGrid, dimBlock>>>(pPartials, (uchar4*)pInput, w);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in first kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }

        //Second kernel: accumulate partial results:
        {
            dim3 dimGrid( 1 );
            dim3 dimBlock( 3*256 );
            cudaEventRecord(evt[2]);
            gpu_histo_accumulate<<<dimGrid, dimBlock>>>(pPartials, nBlocksH, pOutput);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in second kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[3]);
        }
        
        //Calculate time:
        cudaEventSynchronize(evt[3]);
        float dt = 0.0f;//milliseconds
        cudaEventElapsedTime(&dt, evt[0], evt[1]);
        dt1 = dt;
        cudaEventElapsedTime(&dt, evt[2], evt[3]);
        dt1 += dt;
        for(auto& e : evt){ cudaEventDestroy(e); }

        std::vector<unsigned int> tmp(3*256);
        err = cudaMemcpy( tmp.data(), pOutput, 3*256*sizeof(unsigned int), cudaMemcpyDeviceToHost );
        if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pInput );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pPartials );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pOutput );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        gpu1.fromLinearMemory(tmp);
    }

    //GPU version using shared atomics:
    float dt2 = 0.0f;
    {
        unsigned char* pInput    = nullptr;
        unsigned int*  pPartials = nullptr;
        unsigned int*  pOutput   = nullptr;

        cudaError_t err = cudaSuccess;
        err = cudaMalloc( (void**)&pInput, w*h*sizeof(color) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMalloc( (void**)&pPartials, nBlocksH*3*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemset(pPartials, 0, nBlocksH*3*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        err = cudaMalloc( (void**)&pOutput, 3*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemcpy( pInput, data0, w*h*sizeof(color), cudaMemcpyHostToDevice );
        if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

        cudaEvent_t evt[4];
        for(auto& e : evt){ cudaEventCreate(&e); }
        
        //First kernel of global histograms:
        {
            dim3 dimGrid( 1, nBlocksH );
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[0]);
            gpu_histo_shared_atomics<<<dimGrid, dimBlock>>>(pPartials, (uchar4*)pInput, w);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in third kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }

        //Second kernel: accumulate partial results:
        {
            dim3 dimGrid( 1 );
            dim3 dimBlock( 3*256 );
            cudaEventRecord(evt[2]);
            gpu_histo_accumulate<<<dimGrid, dimBlock>>>(pPartials, nBlocksH, pOutput);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in fourth kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[3]);
        }
        
        //Calculate time:
        cudaEventSynchronize(evt[3]);
        float dt = 0.0f;//milliseconds
        cudaEventElapsedTime(&dt, evt[0], evt[1]);
        dt2 = dt;
        cudaEventElapsedTime(&dt, evt[2], evt[3]);
        dt2 += dt;
        for(auto& e : evt){ cudaEventDestroy(e); }

        std::vector<unsigned int> tmp(3*256);
        err = cudaMemcpy( tmp.data(), pOutput, 3*256*sizeof(unsigned int), cudaMemcpyDeviceToHost );
        if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pInput );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pPartials );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pOutput );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        gpu2.fromLinearMemory(tmp);
    }

    
    auto compare = [w, h, &cpu](three_histograms const& histos)
    {
        int mismatches = 0;
        for(int i=0; i<256; ++i)
        {
            if(histos.rh[i] != cpu.rh[i]){ std::cout << "Mismatch: red   at " << i << " : " << histos.rh[i] << " != " << cpu.rh[i] << "\n"; mismatches += 1; }
            if(histos.gh[i] != cpu.gh[i]){ std::cout << "Mismatch: green at " << i << " : " << histos.gh[i] << " != " << cpu.gh[i] << "\n"; mismatches += 1; }
            if(histos.bh[i] != cpu.bh[i]){ std::cout << "Mismatch: blue  at " << i << " : " << histos.bh[i] << " != " << cpu.bh[i] << "\n"; mismatches += 1; }
        }
        return mismatches;
    };

    int mismatches1 = compare(gpu1);
    if     (mismatches1 == 0){ std::cout << "CPU result matches GPU global atomics result.\n"; }
    else if(mismatches1 == 1){ std::cout << "There was 1 mismatch between the CPU and GPU global atomics result.\n"; }
    else                     { std::cout << "There were " << mismatches1 << " mismatches between the CPU and GPU global atomics result.\n"; }

    int mismatches2 = compare(gpu2);
    if     (mismatches2 == 0){ std::cout << "CPU result matches GPU shared atomics result.\n"; }
    else if(mismatches2 == 1){ std::cout << "There was 1 mismatch between the CPU and GPU shared atomics result.\n"; }
    else                     { std::cout << "There were " << mismatches2 << " mismatches between the CPU and GPU shared atomics result.\n"; }

    std::cout << "CPU Computation took:                " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "GPU global atomics computation took: " << dt1  << " ms\n";
    std::cout << "GPU shared atomics computation took: " << dt2 << " ms\n";
    
    auto write_histogram = [](std::string const& filename, three_histograms const& data )
    {
        int w = 800;
        int h = 800;
        std::vector<color> image(w*h);
        color white{255, 255, 255, 255};
        std::fill(image.begin(), image.end(), white);
        auto max_r = *std::max_element(data.rh.begin(), data.rh.end());
        auto max_g = *std::max_element(data.gh.begin(), data.gh.end());
        auto max_b = *std::max_element(data.bh.begin(), data.bh.end());
        auto div = std::max(std::max(max_r, max_g), max_b);

        auto fill_rect = [&](int x0, int y0, int width, int height, color const& c)
        {
            for(int y=y0; y>y0-height; --y)
            {
                for(int x=x0; x<x0+width; ++x)
                {
                    image[y*w+x] = c;
                }
            }
        };

        for(int i=0; i<256; ++i)
        {
            //std::cout << i << "   " << data.rh[i] << " " << data.gh[i] << " " << data.bh[i] << "\n";
            fill_rect(i, 780, 1, data.rh[i]*700/div, color{(unsigned char)i, 0, 0, 255});
            fill_rect(i+256, 780, 1, data.gh[i]*700/div, color{0, (unsigned char)i, 0, 255});
            fill_rect(i+256*2, 780, 1, data.bh[i]*700/div, color{0, 0, (unsigned char)i, 255});
        }
        

        int res = stbi_write_jpg(filename.c_str(), w, h, 4, image.data(), 40);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }else{ std::cout << "Output written to file " << filename << "\n"; }
    };

    write_histogram(output_filename1, cpu);
    write_histogram(output_filename2, gpu1);
    write_histogram(output_filename3, gpu2);

    //free input image
    stbi_image_free(data0);

	return 0;
}
