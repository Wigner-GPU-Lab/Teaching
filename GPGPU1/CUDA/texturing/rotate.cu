#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

struct color{ float r, g, b, a; };

void cpu_rotate( std::vector<color>& output, std::vector<color>const & input, int W, int H, float angle )
{
    for(int y=0; y<H; ++y)
    {
        for(int x=0; x<W; ++x)
        {
            float u = x - W / 2.0f;
            float v = y - H / 2.0f;

            float u0 = u * std::cos(angle) - v * std::sin(angle) + W / 2.0f;
            float v0 = v * std::cos(angle) + u * std::sin(angle) + H / 2.0f;

            int ui0 = (int)u0;
            int ui1 = (int)u0+1;
            int vi0 = (int)v0;
            int vi1 = (int)v0+1;

            color c0 = (ui0 >= 0 && vi0 >= 0 && vi0 < H && ui0 < W) ? input[vi0*W+ui0] : color{0, 0, 0, 0};
            color c1 = (ui0 >= 0 && vi1 >= 0 && vi1 < H && ui0 < W) ? input[vi1*W+ui0] : color{0, 0, 0, 0};
            color c2 = (ui1 >= 0 && vi0 >= 0 && vi0 < H && ui1 < W) ? input[vi0*W+ui1] : color{0, 0, 0, 0};
            color c3 = (ui1 >= 0 && vi1 >= 0 && vi1 < H && ui1 < W) ? input[vi1*W+ui1] : color{0, 0, 0, 0};

            //bilinear interpolation:
            float ufrac = ui0 + 1 - u0;
            float vfrac = vi0 + 1 - v0;

            float Ar = c0.r * ufrac + c2.r * (1 - ufrac);
            float Ag = c0.g * ufrac + c2.g * (1 - ufrac);
            float Ab = c0.b * ufrac + c2.b * (1 - ufrac);
            float Aa = c0.a * ufrac + c2.a * (1 - ufrac);

            float Br = c1.r * ufrac + c3.r * (1 - ufrac);
            float Bg = c1.g * ufrac + c3.g * (1 - ufrac);
            float Bb = c1.b * ufrac + c3.b * (1 - ufrac);
            float Ba = c1.a * ufrac + c3.a * (1 - ufrac);

            float Cr = Ar * vfrac + Br * (1 - vfrac);
            float Cg = Ag * vfrac + Bg * (1 - vfrac);
            float Cb = Ab * vfrac + Bb * (1 - vfrac);
            float Ca = Aa * vfrac + Ba * (1 - vfrac);

            Cr = Cr < 0 ? 0 : Cr;
            Cg = Cg < 0 ? 0 : Cg;
            Cb = Cb < 0 ? 0 : Cb;
            Ca = Ca < 0 ? 0 : Ca;
           
            output[y*W+x] = color{Cr, Cg, Cb, Ca};
        }
    }
}

__global__ void gpu_rotate_buffer( float4* output, float4* input, int W, int H, float angle )
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    float u = x - W / 2.0f;
    float v = y - H / 2.0f;

    float u0 = u * std::cos(angle) - v * std::sin(angle) + W / 2.0f;
    float v0 = v * std::cos(angle) + u * std::sin(angle) + H / 2.0f;

    int ui0 = (int)u0;
    int ui1 = (int)u0+1;
    int vi0 = (int)v0;
    int vi1 = (int)v0+1;

    float4 c0 = (ui0 >= 0 && vi0 >= 0 && vi0 < H && ui0 < W) ? input[vi0*W+ui0] : float4{0, 0, 0, 0};
    float4 c1 = (ui0 >= 0 && vi1 >= 0 && vi1 < H && ui0 < W) ? input[vi1*W+ui0] : float4{0, 0, 0, 0};
    float4 c2 = (ui1 >= 0 && vi0 >= 0 && vi0 < H && ui1 < W) ? input[vi0*W+ui1] : float4{0, 0, 0, 0};
    float4 c3 = (ui1 >= 0 && vi1 >= 0 && vi1 < H && ui1 < W) ? input[vi1*W+ui1] : float4{0, 0, 0, 0};

    //bilinear interpolation:
    float ufrac = ui0 + 1 - u0;
    float vfrac = vi0 + 1 - v0;

    float Ar = c0.x * ufrac + c2.x * (1 - ufrac);
    float Ag = c0.y * ufrac + c2.y * (1 - ufrac);
    float Ab = c0.z * ufrac + c2.z * (1 - ufrac);
    float Aa = c0.w * ufrac + c2.w * (1 - ufrac);

    float Br = c1.x * ufrac + c3.x * (1 - ufrac);
    float Bg = c1.y * ufrac + c3.y * (1 - ufrac);
    float Bb = c1.z * ufrac + c3.z * (1 - ufrac);
    float Ba = c1.w * ufrac + c3.w * (1 - ufrac);

    float Cr = Ar * vfrac + Br * (1 - vfrac);
    float Cg = Ag * vfrac + Bg * (1 - vfrac);
    float Cb = Ab * vfrac + Bb * (1 - vfrac);
    float Ca = Aa * vfrac + Ba * (1 - vfrac);

    Cr = Cr < 0 ? 0 : Cr;
    Cg = Cg < 0 ? 0 : Cg;
    Cb = Cb < 0 ? 0 : Cb;
    Ca = Ca < 0 ? 0 : Ca;
    
    output[y*W+x] = float4{Cr, Cg, Cb, Ca};
}

__global__ void gpu_rotate_texture( float4* output, cudaTextureObject_t input, int W, int H, float angle )
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x - W/2.0f;
    float v = y - H/2.0f;

    float u0 = u * std::cos(angle) - v * std::sin(angle) + W / 2.0f + 0.5f;
    float v0 = v * std::cos(angle) + u * std::sin(angle) + H / 2.0f + 0.5f;

    // Read from texture and write to global memory
    output[y*W+x] = tex2D<float4>(input, u0, v0);
}

int main()
{
    static const std::string input_filename   = "map.png";
    static const std::string output_filename1 = "cpu_out.jpg";
    static const std::string output_filename2 = "gpu_out1.jpg";
    static const std::string output_filename3 = "gpu_out2.jpg";

    static const int block_size = 32;
    
    /*std::cout << "Enter rotation angle in degrees:\n";
    float angle = 0.5f;
    std::cin >> angle;
    angle *= 3.1415926535f / 180.0f;*/

    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components

    struct rawcolor{ unsigned char r, g, b, a; };

    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }

    std::vector<color> input  (w*h);
    std::vector<color> output1(w*h);
    std::vector<color> output2(w*h);
    std::vector<color> output3(w*h);

    std::transform(data0, data0+w*h, input.begin(), [](rawcolor c){ return color{c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f}; } );
    stbi_image_free(data0);
   
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_rotate(output1, input, w, h, angle);
    auto t1 = std::chrono::high_resolution_clock::now();

    //GPU version using buffers:
    std::array<float, 360> dt1s;
    {
        float4* pInput = nullptr;
        float4* pOutput = nullptr;

        cudaEvent_t evt[2];
        

        cudaError_t err = cudaSuccess;
        err = cudaMalloc( (void**)&pInput, w*h*sizeof(color) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        err = cudaMalloc( (void**)&pOutput, w*h*sizeof(color) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemcpy( pInput, input.data(), w*h*sizeof(color), cudaMemcpyHostToDevice );
        if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

        dim3 dimGrid( w / block_size, h / block_size );
        dim3 dimBlock( block_size, block_size );
        for(int a=0; a<360; ++a)
        {
            for(auto& e : evt){ cudaEventCreate(&e); }
            
            cudaEventRecord(evt[0]);
            gpu_rotate_buffer<<<dimGrid, dimBlock>>>(pOutput, pInput, w, h, a/180.0f*3.1415926535f);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in buffered kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
            cudaEventSynchronize(evt[1]);
            float dt = 0.0f;//milliseconds
            cudaEventElapsedTime(&dt, evt[0], evt[1]);
            for(auto& e : evt){ cudaEventDestroy(e); }
            dt1s[a] = dt;
        }

        err = cudaMemcpy( output2.data(), pOutput, w*h*sizeof(color), cudaMemcpyDeviceToHost );
        if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pInput );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pOutput );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    //GPU version using textures:
    std::array<float, 360> dt2s;
    {
        cudaError_t err = cudaSuccess;
        cudaEvent_t evt[2];
        
        //Channel layout of data:
        cudaChannelFormatDesc channelDescInput  = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
       
        //Allocate data:
        cudaArray* aInput;
        
        err = cudaMallocArray(&aInput, &channelDescInput, w, h);
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        //Upload data to device:
        err = cudaMemcpyToArray(aInput,  0, 0, input.data(), w*h*sizeof(color), cudaMemcpyHostToDevice);
        if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        //Specify texture resource description:
        cudaResourceDesc resdescInput{};
        resdescInput.resType = cudaResourceTypeArray;
        resdescInput.res.array.array = aInput;

        //Specify texture description:
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0]   = cudaAddressModeBorder;
        texDesc.addressMode[1]   = cudaAddressModeBorder;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaTextureObject_t texObjInput = 0;
        err = cudaCreateTextureObject(&texObjInput,  &resdescInput,  &texDesc, nullptr);
        if( err != cudaSuccess){ std::cout << "Error creating texture object: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        //The output is just an usual buffer:
        float4* pOutput = nullptr;
        err = cudaMalloc( (void**)&pOutput, w*h*sizeof(color) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        dim3 dimGrid( w / block_size, h / block_size );
        dim3 dimBlock( block_size, block_size );
        // Invoke kernel
        for(int a=0; a<360; ++a)
        {
            for(auto& e : evt){ cudaEventCreate(&e); }
            cudaEventRecord(evt[0]);
            gpu_rotate_texture<<<dimGrid, dimBlock>>>(pOutput, texObjInput, w, h, a/180.0f*3.1415926535f);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
            cudaEventSynchronize(evt[1]);
            float dt = 0.0f;//milliseconds
            cudaEventElapsedTime(&dt, evt[0], evt[1]);
            for(auto& e : evt){ cudaEventDestroy(e); }
            dt2s[a] = dt;
        }

        err = cudaMemcpy( output3.data(), pOutput, w*h*sizeof(color), cudaMemcpyDeviceToHost );
        if( err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

        //Cleanup:
        err = cudaDestroyTextureObject(texObjInput);
        if( err != cudaSuccess){ std::cout << "Error destroying texture object: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFreeArray( aInput );
        if( err != cudaSuccess){ std::cout << "Error freeing array allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree( pOutput );
        if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        
    }

    auto diff = [](float x, float y){ return std::abs(x-y) > 1.0f/255.0f; };
    auto conv = [](float x){ return (int)(x*255.0f); };
    auto compare = [w, h, &output1, diff, conv](std::vector<color> const& output)
    {
        int mismatches = 0;
        for(int y=0; y<h; ++y)
        {
            for(int x=0; x<w; ++x)
            {
                color cpu = output1[y*w+x];
                color gpu = output [y*w+x];
                if( diff(cpu.r, gpu.r) || diff(cpu.g, gpu.g) || diff(cpu.b, gpu.b) || diff(cpu.a, gpu.a) )
                {
                    mismatches += 1;
                    /*std::cout << "Difference at pixel: (" << x << ", " << y << ") :" <<
                    " r=" << conv(cpu.r) << " " << conv(gpu.r) <<
                    " g=" << conv(cpu.g) << " " << conv(gpu.g) <<
                    " b=" << conv(cpu.b) << " " << conv(gpu.b) <<
                    " a=" << conv(cpu.a) << " " << conv(gpu.a) << "\n";*/
                }
            }   
        }
        return mismatches;
    };

    int mismatches1 = compare(output2);
    if     (mismatches1 == 0){ std::cout << "CPU result matches buffered GPU result.\n"; }
    else if(mismatches1 == 1){ std::cout << "There was 1 mismatch between the CPU and buffered GPU result.\n"; }
    else                     { std::cout << "There were " << mismatches1 << " mismatches between the CPU and buffered GPU result.\n"; }

    int mismatches2 = compare(output3);
    if     (mismatches2 == 0){ std::cout << "CPU result matches textured GPU result.\n"; }
    else if(mismatches2 == 1){ std::cout << "There was 1 mismatch between the CPU and textured GPU result.\n"; }
    else                     { std::cout << "There were " << mismatches2 << " mismatches between the CPU and textured GPU result.\n"; }

    std::cout << "CPU Computation took:          " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";
    std::cout << "Buffered GPU Computation took: " << dt  << " ms\n";
    std::cout << "Textured GPU Computation took: " << dt2 << " ms\n";
    std::cout << "Textured / Buffered improvement : " << (int)(100-dt2/dt*100) << " %\n";

    
    auto convert_and_write = [w, h, ch](std::string const& filename, std::vector<color> const& data )
    {
        std::vector<rawcolor> tmp(w*h*ch);
        std::transform(data.cbegin(), data.cend(), tmp.begin(),
            [](color c){ return rawcolor{   (unsigned char)(c.r*255.0f),
                                            (unsigned char)(c.g*255.0f),
                                            (unsigned char)(c.b*255.0f),
                                            (unsigned char)(c.a*255.0f) }; } );

        int res = stbi_write_jpg(filename.c_str(), w, h, ch, tmp.data(), 40);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }else{ std::cout << "Output written to file " << filename << "\n"; }
    };

    convert_and_write(output_filename1, output1);
    convert_and_write(output_filename2, output2);
    convert_and_write(output_filename3, output3);

	return 0;
}
